import cv2 as cv
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_video_skeleton(input_folder: str = None, output_folder: str = None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of videos for the progress bar
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.avi')]
    print(f"Starting extraction for {len(video_files)} videos...")

    for filename in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(input_folder, filename)
        cap = cv.VideoCapture(video_path)
        skeleton_data = []

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("End of video")
                break

            # Convert the BGR image to RGB
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Process the image and detect the pose
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # Get x, y, z for all 33 landmarks
                current_frame = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
            else:
                # If no landmarks detected, append zeros
                current_frame = np.zeros((33, 4))
            
            skeleton_data.append(current_frame)

        cap.release()

        # Convert to Numpy array: shape (Frames, 33, 4)
        video_array = np.array(skeleton_data)

        # Save file with the same name, but .npy extension
        save_path = os.path.join(output_folder, filename.replace('.avi', '.npy'))
        np.save(save_path, video_array)
        print(f"Processed: {filename} -> Shape: {video_array.shape}")

if __name__ == "__main__":
    for i in range(7, 11):
        index = str(i).zfill(2)
        video_folder = rf"/media/phonght/New Volume/NTU-dataset/rgb-videos/nturgbd_rgb_s0{index}_single_actor/nturgb+d_rgb"
        output_folder = rf"/media/phonght/New Volume/NTU-dataset/mp_skeletons/nturgbd_rgb_s0{index}_single_actor/nturgb+d_skeleton"
        process_video_skeleton(video_folder, output_folder)