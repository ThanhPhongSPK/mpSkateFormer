import numpy as np
import os
import cv2 as cv

def load_skeleton_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    num_frames = int(lines[0])
    line_idx = 1
    all_frames = []

    for _ in range(num_frames):
        num_bodies = int(lines[line_idx])
        line_idx += 1
        frame_joints = []
        for _ in range(num_bodies):
            bodyID = lines[line_idx].strip().split()[0]
            line_idx += 1
            num_joints = int(lines[line_idx])
            line_idx += 1

            joints_xyz = []
            joints_uv = []
            for _ in range(num_joints):
                data = lines[line_idx].split()
                xyz = list(map(float, data[0:3]))
                uv = list(map(float, data[5:7]))
                joints_xyz.append(xyz)
                joints_uv.append(uv)
                line_idx += 1
            
            frame_joints.append({
                'bodyID': bodyID,
                'joints_xyz': np.array(joints_xyz),
                'joints_uv': np.array(joints_uv)
            })
        all_frames.append(frame_joints)

    return all_frames

if __name__ == "__main__":
    base_skeleton_dir = "/media/phonght/New Volume/NTU-dataset/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/"
    skeleton_path = os.path.join(base_skeleton_dir, "S001C001P001R001A001.skeleton")

    base_video_dir = "/media/phonght/New Volume/NTU-dataset/rgb-videos/nturgbd_rgb_s001/nturgb+d_rgb"
    video_path = os.path.join(base_video_dir, "S001C001P001R001A001_rgb.avi")

    all_frames = load_skeleton_file(skeleton_path)

    for frame_idx in range(min(20, len(all_frames))):
        print(f"\n=== Frame {frame_idx} ===")
        for body_idx, body in enumerate(all_frames[frame_idx]):
            print(f"Body {body_idx} (ID: {body['bodyID']}):")
            print(body['joints_xyz'])

