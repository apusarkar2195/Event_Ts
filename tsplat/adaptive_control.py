import torch
import torch.nn as nn 


#========================================================================
# GENERAL UTIL FUNCTIONS
#========================================================================
def scaled_sigmoid(x):
    return 5 * torch.sigmoid(x)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def inverse_sigmoid_10(x):
    return -torch.log((10 / x) - 1)

opacity_activation = torch.sigmoid
inverse_opacity_activation = inverse_sigmoid
exponential_activation = lambda x: 0.01 + torch.exp(x)
inverse_exponential_activation = lambda y: torch.log(y - 0.01)
#==========================================================================



def sample_alives(probs, num, big_mask, alive_indices=None):
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=0.0)
    probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
    sampled_idxs = torch.multinomial(probs, min(num, (probs>0).sum().item()), replacement=False)

    if alive_indices is not None:
        sampled_idxs = alive_indices[sampled_idxs]

    device = probs.device
    cost_true = torch.tensor(3, dtype=torch.int64, device=device)
    cost_false = torch.tensor(1, dtype=torch.int64, device=device)
    costs = torch.where(big_mask[sampled_idxs], cost_true, cost_false)
    
    cum_costs = torch.cumsum(costs, dim=0)
    
    cutoff_idx = (cum_costs >= num).nonzero(as_tuple=True)[0]
    if cutoff_idx.numel() > 0:
        cutoff = cutoff_idx[0].item() + 1 
    else:
        cutoff = sampled_idxs.numel()
    return sampled_idxs[:cutoff]

def prune_optimizer(optimizer, mask):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def cat_tensors_to_optimizer(optimizer, tensors_dict):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        extension_tensor = tensors_dict[group["name"]]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:

            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
    

    return optimizable_tensors


def replace_tensors_to_optimizer(splat, optimizer, inds=None):
    tensors_dict = {"triangles_points": splat.triangles_points,
        "sh0": splat.sh0,
        "shN": splat.shN,
        "opacity": splat.opacity,
        "sigma" : splat.sigma,
        "mask": splat.mask}

    optimizable_tensors = {}
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        tensor = tensors_dict[group["name"]]
        stored_state = optimizer.state.get(group['params'][0], None)
        
        if inds is not None:
            stored_state["exp_avg"][inds] = 0
            stored_state["exp_avg_sq"][inds] = 0
        else:
            stored_state["exp_avg"] = torch.zeros_like(tensor)
            stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

        del optimizer.state[group['params'][0]]
        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state

        optimizable_tensors[group["name"]] = group["params"][0]

    splat.triangles_points = optimizable_tensors["triangles_points"]
    splat.sh0 =  optimizable_tensors["sh0"]
    splat.shN = optimizable_tensors["shN"]
    splat.opacity = optimizable_tensors["opacity"]
    splat.sigma = optimizable_tensors["sigma"] 
    splat.mask = optimizable_tensors["mask"]

    torch.cuda.empty_cache()
    
    return optimizable_tensors


def densification_postfix(splat, splat_non_learnable, optimizer, new_triangles_points, new_sh0, new_shN, new_opacities, new_sigma, new_mask):
    d = {"triangles_points": new_triangles_points,
    "sh0": new_sh0,
    "shN": new_shN,
    "opacity": new_opacities,
    "sigma" : new_sigma,
    "mask": new_mask}

    optimizable_tensors = cat_tensors_to_optimizer(optimizer, d)
    splat.triangles_points = optimizable_tensors["triangles_points"]
    splat.sh0 = optimizable_tensors["sh0"]
    splat.shN = optimizable_tensors["shN"]
    splat.opacity = optimizable_tensors["opacity"]
    splat.sigma = optimizable_tensors["sigma"]
    splat.mask = optimizable_tensors["mask"]

    splat_non_learnable["denom"] = torch.zeros((splat.triangles_points.shape[0], 1), device="cuda")
    splat_non_learnable["max_radii2D"] = torch.zeros((splat.triangles_points.shape[0]), device="cuda")
    splat_non_learnable["max_density_factor"] = torch.zeros((splat.triangles_points.shape[0]), device="cuda")
    splat_non_learnable["triangle_area"] = torch.zeros((splat.triangles_points.shape[0]), device="cuda")

    splat_non_learnable["max_scaling"] = torch.cat((splat_non_learnable["max_scaling"], torch.zeros(new_opacities.shape[0], device="cuda")),dim=0)

    num_points_per_triangle = []
    for i in range(splat.triangles_points.size(0)):
        num_points_per_triangle.append(splat.triangles_points[i].shape[0])
    tensor_num_points_per_triangle = torch.tensor(num_points_per_triangle, dtype=torch.int, device='cuda:0')
    cumsum_of_points_per_triangle = torch.cumsum(torch.nn.functional.pad(tensor_num_points_per_triangle, (1,0), value=0), 0, dtype=torch.int)[:-1]
    number_of_points = splat.triangles_points.shape[0]

    splat_non_learnable["num_points_per_triangle"] = tensor_num_points_per_triangle
    splat_non_learnable["cumsum_of_points_per_triangle"] = cumsum_of_points_per_triangle
    splat_non_learnable["number_of_points"] = number_of_points


def update_params(splat, splat_non_learnable, selected_indices):
    selected_triangles_points = splat.triangles_points[selected_indices] # TODO : Need to change

    A = selected_triangles_points[:, 0, :]
    B = selected_triangles_points[:, 1, :]
    C = selected_triangles_points[:, 2, :]

    M_AB = (A + B) / 2
    M_AC = (A + C) / 2
    M_BC = (B + C) / 2

    sub1 = torch.stack([A, M_AB, M_AC], dim=1)
    sub2 = torch.stack([B, M_AB, M_BC], dim=1)
    sub3 = torch.stack([C, M_AC, M_BC], dim=1)
    sub4 = torch.stack([M_AB, M_AC, M_BC], dim=1)

    new_triangles_points = torch.cat([sub1, sub2, sub3, sub4], dim=0)

    # TODO : Need to change accordingly every line : Partially Done
    new_features_dc = splat.sh0[selected_indices].repeat(4, 1, 1)
    new_features_rest = splat.shN[selected_indices].repeat(4, 1, 1)
    new_opacities = splat.opacity[selected_indices].repeat(4, 1)
    new_sigma = splat.sigma[selected_indices].repeat(4, 1)
    new_mask = torch.ones_like(splat.mask[selected_indices].repeat(4, 1))

    return new_triangles_points, new_features_dc, new_features_rest, new_opacities, new_sigma, new_mask

def update_params_small(splat, splat_non_learnable, idxs):
    new_triangles_points = splat.triangles_points[idxs] # TODO : Need to change
    n = new_triangles_points.shape[0]

    v1 = new_triangles_points[:, 1] - new_triangles_points[:, 0]
    v2 = new_triangles_points[:, 2] - new_triangles_points[:, 0]
    normals = torch.cross(v1, v2, dim=1)                  
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-9) 

    min_coords = new_triangles_points.min(dim=1).values
    max_coords = new_triangles_points.max(dim=1).values
    shape_sizes = max_coords - min_coords

    max_noise_factor = splat_non_learnable["max_noise_factor"] # TODO : Need to change max_noise_factor = 1.5
    noise_scale = shape_sizes * max_noise_factor
    noise = (torch.rand(n, 1, 3, device=new_triangles_points.device) - 0.5) * noise_scale.unsqueeze(1)

    dot_products = (noise * normals.unsqueeze(1)).sum(dim=-1, keepdim=True)
    noise_in_plane = noise - dot_products * normals.unsqueeze(1)

    new_triangles_points_noisy = new_triangles_points + noise_in_plane

    opacity_old = opacity_activation(splat.opacity)[idxs] # TODO : Need to change 
    opacity_new = inverse_sigmoid(1.0 - torch.pow(1.0 - opacity_old, 1.0 / 2))
    
    # TODO : Need to change
    return (torch.cat([splat.triangles_points[idxs], new_triangles_points_noisy], dim=0),
            torch.cat([splat.sh0[idxs], splat.sh0[idxs][idxs]], dim=0),
            torch.cat([splat.shN[idxs], splat.shN[idxs]], dim=0),
            torch.cat([opacity_new, opacity_new], dim=0),
            torch.cat([splat.sigma[idxs], splat.sigma[idxs]], dim=0),
            torch.cat([splat.mask[idxs], torch.ones((n, 1), device=splat.mask.device)], dim=0))

def prune_points(splat, splat_non_learnable, optimizer, mask):
    valid_points_mask = ~mask
    optimizable_tensors = prune_optimizer(optimizer, valid_points_mask)

    splat.triangles_points = optimizable_tensors["triangles_points"]
    splat.sh0 = optimizable_tensors["sh0"]
    splat.shN = optimizable_tensors["shN"]
    splat.opacity = optimizable_tensors["opacity"]
    splat.sigma = optimizable_tensors["sigma"]
    splat.mask = optimizable_tensors["mask"]

    num_points_per_triangle = []
    for i in range(splat.triangles_points.size(0)):
        num_points_per_triangle.append(splat.triangles_points[i].shape[0])
    tensor_num_points_per_triangle = torch.tensor(num_points_per_triangle, dtype=torch.int, device='cuda:0')
    cumsum_of_points_per_triangle = torch.cumsum(torch.nn.functional.pad(tensor_num_points_per_triangle, (1,0), value=0), 0, dtype=torch.int)[:-1]
    number_of_points = splat.triangles_points.shape[0]

    splat_non_learnable["num_points_per_triangle"] = tensor_num_points_per_triangle
    splat_non_learnable["cumsum_of_points_per_triangle"] = cumsum_of_points_per_triangle
    splat_non_learnable["number_of_points"] = number_of_points


def add_new_triangles_without_removing(splat, splat_non_learnable, optimizer, cap_max, oddGroup=True, dead_mask=None):
    current_num_points = splat.opacity.shape[0]
    target_num = min(cap_max, int(splat_non_learnable["add_shape"] * current_num_points))
    num_gs = max(0, target_num - current_num_points)

    num_gs += dead_mask.sum()

    if num_gs <= 0:
        return 0

    if oddGroup:
        probs = opacity_activation(splat.opacity).squeeze(-1) 
    else:
        eps = torch.finfo(torch.float32).eps
        probs = exponential_activation(splat.sigma).squeeze(-1) 
        probs = 1 / (probs + eps)
        
    probs[dead_mask] = 0

    compar = splat_non_learnable["image_size"]
    big_mask   = compar > splat_non_learnable["split_size"]

    add_idx = sample_alives(probs=probs, num=num_gs, big_mask=big_mask)

    big_mask   = compar[add_idx] > splat_non_learnable["split_size"]
    small_mask = ~big_mask
    big_indices   = add_idx[big_mask]
    small_indices = add_idx[small_mask]

    num_big = big_indices.shape[0]
    if num_big > 0:

        (split_triangles_points,
        split_sh0,
        split_shN,
        split_opacity,
        split_sigma,
        split_mask) = update_params(splat, splat_non_learnable, big_indices)

    else:
        split_triangles_points  = torch.empty((0, 3, 3),   device=splat.triangles_points.device)
        split_sh0    = torch.empty((0,) + splat.sh0.shape[1:],   device=splat.sh0.device) # TODO : need to fix sh0 dim = N,0:1,K where as feate_dc shape = N,3,0:1
        split_shN  = torch.empty((0,) + splat.shN.shape[1:], device=splat.shN.device) # TODO : need to fix shN dim = N,1:,3 where as feate_rest shape = N,3,1:
        split_opacity        = torch.empty((0, 1), device=splat.opacity.device)
        split_sigma          = torch.empty((0, 1), device=splat.sigma.device)
        split_mask           = torch.empty((0, 1), device=splat.mask.device)


    num_small = small_indices.shape[0]
    if num_small > 0:
        (clone_triangles_points,
        clone_sh0,
        clone_shN,
        clone_opacity,
        clone_sigma,
        clone_mask) = update_params_small(splat, splat_non_learnable, small_indices)

    else:
        clone_triangles_points  = torch.empty((0, 3, 3),   device=splat.triangles_points.device)
        clone_sh0    = torch.empty((0,) + splat.sh0.shape[1:],   device=splat.sh0.device)
        clone_shN  = torch.empty((0,) + splat.shN.shape[1:], device=splat.shN.device)
        clone_opacity        = torch.empty((0, 1), device=splat.opacity.device)
        clone_sigma          = torch.empty((0, 1), device=splat.sigma.device)
        clone_mask           = torch.empty((0, 1), device=splat.mask.device)

    new_triangles_points = torch.cat([split_triangles_points, clone_triangles_points], dim=0)
    #==========================
    print('**********************************************************************')
    print(f'New triangle points to be added : {new_triangles_points.shape[0]}')
    print(f"Number of trinangles to be removed : {len(add_idx)}")
    print('***********************************************************************')
    #==========================
    new_sh0   = torch.cat([split_sh0,   clone_sh0],   dim=0)
    new_shN = torch.cat([split_shN, clone_shN], dim=0)
    new_opacity       = torch.cat([split_opacity,       clone_opacity],       dim=0)
    new_sigma         = torch.cat([split_sigma,         clone_sigma],         dim=0)
    new_mask          = torch.cat([split_mask,          clone_mask],          dim=0)

    densification_postfix(splat, splat_non_learnable, optimizer, new_triangles_points, new_sh0, new_shN, new_opacity, new_sigma, new_mask)
    replace_tensors_to_optimizer(splat, optimizer, inds=add_idx)

    mask = torch.zeros(splat.opacity.shape[0], dtype=torch.bool)
    mask[add_idx] = True
    mask[torch.nonzero(dead_mask, as_tuple=True)] = True
    prune_points(splat, splat_non_learnable, optimizer, mask)

    splat_non_learnable["triangle_area"] = torch.zeros((splat.triangles_points.shape[0]), device="cuda")
    splat_non_learnable["image_size"] = torch.zeros((splat.triangles_points.shape[0]), device="cuda")
    splat_non_learnable["importance_score"] = torch.zeros((splat.triangles_points.shape[0]), device="cuda")

def add_new_triangles(splat, splat_non_learnable, optimizer, cap_max, oddGroup=True, dead_mask=None):
    current_num_points = splat.opacity.shape[0]
    target_num = min(cap_max, int(splat_non_learnable["add_shape"] * current_num_points))
    num_gs = max(0, target_num - current_num_points)

    num_gs += dead_mask.sum()

    if num_gs <= 0:
        return 0

    if oddGroup:
        probs = opacity_activation(splat.opacity).squeeze(-1) 
    else:
        eps = torch.finfo(torch.float32).eps
        probs = exponential_activation(splat.sigma).squeeze(-1) 
        probs = 1 / (probs + eps)
        
    probs[dead_mask] = 0

    compar = splat_non_learnable["image_size"]
    big_mask   = compar > splat_non_learnable["split_size"]

    add_idx = sample_alives(probs=probs, num=num_gs, big_mask=big_mask)

    big_mask   = compar[add_idx] > splat_non_learnable["split_size"]
    small_mask = ~big_mask
    big_indices   = add_idx[big_mask]
    small_indices = add_idx[small_mask]

    num_big = big_indices.shape[0]
    if num_big > 0:

        (split_triangles_points,
        split_sh0,
        split_shN,
        split_opacity,
        split_sigma,
        split_mask) = update_params(splat, splat_non_learnable, big_indices)

    else:
        split_triangles_points  = torch.empty((0, 3, 3),   device=splat.triangles_points.device)
        split_sh0    = torch.empty((0,) + splat.sh0.shape[1:],   device=splat.sh0.device) # TODO : need to fix sh0 dim = N,0:1,K where as feate_dc shape = N,3,0:1
        split_shN  = torch.empty((0,) + splat.shN.shape[1:], device=splat.shN.device) # TODO : need to fix shN dim = N,1:,3 where as feate_rest shape = N,3,1:
        split_opacity        = torch.empty((0, 1), device=splat.opacity.device)
        split_sigma          = torch.empty((0, 1), device=splat.sigma.device)
        split_mask           = torch.empty((0, 1), device=splat.mask.device)


    num_small = small_indices.shape[0]
    if num_small > 0:
        (clone_triangles_points,
        clone_sh0,
        clone_shN,
        clone_opacity,
        clone_sigma,
        clone_mask) = update_params_small(splat, splat_non_learnable, small_indices)

    else:
        clone_triangles_points  = torch.empty((0, 3, 3),   device=splat.triangles_points.device)
        clone_sh0    = torch.empty((0,) + splat.sh0.shape[1:],   device=splat.sh0.device)
        clone_shN  = torch.empty((0,) + splat.shN.shape[1:], device=splat.shN.device)
        clone_opacity        = torch.empty((0, 1), device=splat.opacity.device)
        clone_sigma          = torch.empty((0, 1), device=splat.sigma.device)
        clone_mask           = torch.empty((0, 1), device=splat.mask.device)

    new_triangles_points = torch.cat([split_triangles_points, clone_triangles_points], dim=0)
    #==========================
    print('**********************************************************************')
    print(f'New triangle points to be added : {new_triangles_points.shape[0]}')
    print(f"Number of trinangles to be removed : {len(add_idx)}")
    print('***********************************************************************')
    #==========================
    new_sh0   = torch.cat([split_sh0,   clone_sh0],   dim=0)
    new_shN = torch.cat([split_shN, clone_shN], dim=0)
    new_opacity       = torch.cat([split_opacity,       clone_opacity],       dim=0)
    new_sigma         = torch.cat([split_sigma,         clone_sigma],         dim=0)
    new_mask          = torch.cat([split_mask,          clone_mask],          dim=0)

    densification_postfix(splat, splat_non_learnable, optimizer, new_triangles_points, new_sh0, new_shN, new_opacity, new_sigma, new_mask)
    replace_tensors_to_optimizer(splat, optimizer, inds=add_idx)

    mask = torch.zeros(splat.opacity.shape[0], dtype=torch.bool)
    mask[add_idx] = True
    mask[torch.nonzero(dead_mask, as_tuple=True)] = True
    prune_points(splat, splat_non_learnable, optimizer, mask)

    splat_non_learnable["triangle_area"] = torch.zeros((splat.triangles_points.shape[0]), device="cuda")
    splat_non_learnable["image_size"] = torch.zeros((splat.triangles_points.shape[0]), device="cuda")
    splat_non_learnable["importance_score"] = torch.zeros((splat.triangles_points.shape[0]), device="cuda")    

def remove_final_points(splat, splat_non_learnable, mask):
    prune_points(mask)
    splat_non_learnable["triangle_area"] = torch.zeros((splat.triangles_points.shape[0]), device="cuda")
    splat_non_learnable["image_size"] = torch.zeros((splat.triangles_points.shape[0]), device="cuda")
    splat_non_learnable["importance_score"] = torch.zeros((splat.triangles_points.shape[0]), device="cuda")    
