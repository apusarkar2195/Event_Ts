# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="Bingxin/Marigold",
#     revision="0f4e0021da7a153804301f8b988e8b7b4daf056b",
#     local_dir="/media/vplab/ExtremeSSD1/Apu/Marigold_Model",   # pick any permanent folder
#     local_dir_use_symlinks=False                  # ensures real files, not symlinks to cache
# )

import numpy as np
import os

# event_path = "/media/vplab/ExtremeSSD1/Apu/3D_Event/DataSet/event_replica/room0/event_threshold_0.1"
# event_path = os.path.join(event_path, 'gray_events_data.npy')
# raw_events = np.load(event_path)
# t_lower = 0.1 - 0.05
# t_upper = 98.95 + 0.05
# print(f"Raw event shape : {raw_events.shape}")
# print(f"Single event : {raw_events[0]}")
# index_lower_bound = raw_events[:, 2] >= t_lower
# index_upper_bound = raw_events[:, 2] < t_upper
# selected_idx = index_lower_bound*index_upper_bound
# selcted_events = raw_events[selected_idx, :]

# print(f"selected event shape : {selcted_events.shape}")

pose_path = "/media/vplab/ExtremeSSD1/Apu/3D_Event/DataSet/event_replica/room0/poses_ts.txt"
pose_ts = np.loadtxt(pose_path)
print(pose_ts.shape)