import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

sys.path.insert(0, '/mnt/A87E3F8E7E3F5472/CapstoneProject')

from my_src.utils.mediapipe2ntu_mapping import MediaPipeToNTUConverter

def visualize_skeleton_3d(model_asset_path, video_path):
    # Initialize converter
    converter = MediaPipeToNTUConverter()
    ntu_edges = MediaPipeToNTUConverter.get_connections()

    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_asset_path),
        running_mode=VisionRunningMode.VIDEO,
    )

    pose_landmarker = PoseLandmarker.create_from_options(options)

    # Open video
    cap = cv.VideoCapture(video_path)
    frame_count = 0

    # Setup matplotlib
    plt.ion()
    fig = plt.figure(figsize=(16, 6))

    print("Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = pose_landmarker.detect_for_video(mp_image, frame_count)
        frame_count += 1
        print(f"Processing frame {frame_count}")

        if detection_result.pose_world_landmarks:
            # Convert to NTU format
            landmarks = detection_result.pose_world_landmarks[0]
            ntu_skeleton = converter.convert(landmarks)

            # Clear and redraw
            fig.clear()

            # Left subplot: Original frame
            ax1 = fig.add_subplot(121)
            ax1.imshow(rgb_frame)
            ax1.set_title(f'Original Video - Frame {frame_count}')
            ax1.axis('off')

            # Right subplot: 3D NTU skeleton
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(ntu_skeleton[:, 0],
                       ntu_skeleton[:, 1],
                       ntu_skeleton[:, 2],
                       c='r', s=100, marker='o', alpha=0.8)
            
            # Plot bones
            for start, end in ntu_edges:
                if start < 24 and end < 24:
                    ax2.plot(
                        [ntu_skeleton[start, 0], ntu_skeleton[end, 0]],
                        [ntu_skeleton[start, 1], ntu_skeleton[end, 1]],
                        [ntu_skeleton[start, 2], ntu_skeleton[end, 2]],
                        'b-', linewidth=2
                    )
            
            # Set lables
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_title(f'3D NTU Skeleton Visualization {frame_count}')

            # Set viewing angle
            ax2.view_init(elev=20, azim=45)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)

        # Check for quit
        if plt.waitforbuttonpress(timeout=0.001):
            break
        
    cap.release()
    cv.destroyAllWindows()
    plt.close('all')


if __name__ == "__main__":
    model_asset_path = './my_src/pretrained/pose_landmarks/pose_landmarker_full.task'  # Update with actual path
    video_path = '/mnt/A87E3F8E7E3F5472/CapstoneProject/my_src/video/S001C001P001R001A001_rgb.avi'  # Update with actual path
    visualize_skeleton_3d(model_asset_path, video_path)
