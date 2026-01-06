import sys
import torch
import cv2 as cv
import numpy as np
import os
import sys
import time
import mediapipe as mp
from collections import deque

sys.path.insert(0, '/mnt/A87E3F8E7E3F5472/CapstoneProject')

from my_src.models.SkateFormer import create_mediapipe_skateformer
from my_src.utils.dataloader import NTU60_CLASSES

# CONFIG
CHECKPOINT_PATH = "my_src/results/20251229_010500_final/best_skateformer_model.pth"
INPUT_VIDEO_PATH = r"my_src/video/Test_video.mp4"
OUTPUT_VIDEO_PATH = r""
NUM_CLASSES = 60
FRAMES_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RealTimeProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Sliding Window Buffer: Automatically removes old items when full
        self.skeleton_buffer = deque(maxlen=FRAMES_LEN)

        # Mapping from 33 joints to 36 joints
        self.new_order = [
            11, 13, 15, 17, # Left Arm
            12, 14, 16, 18, # Right Arm
            19, 21, 23, -1, # Left Hand + Hip + Pad
            20, 22, 24, -1, # Right Hand + Hip + Pad
            25, 27, 29, 31, # Left Leg
            26, 28, 30, 32, # Right Leg
            0, 1, 4, -1,    # Face Core + Pad
            2, 3, 7, -1,    # Face Left + Pad
            5, 6, 8, -1     # Face Right + Pad
        ] # -1 indicates padding

    def process_frame(self, frame_rgb):
        """Extracts raw 33 landmarks from a single frame"""
        result = self.pose.process(frame_rgb)
        if result.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark])
            self.skeleton_buffer.append(landmarks)
            return result.pose_landmarks
        else:
            # If detection fails, duplicate last frame or add zeros to keep time moving
            if len(self.skeleton_buffer) > 0:
                self.skeleton_buffer.append(self.skeleton_buffer[-1])
            else:
                self.skeleton_buffer.append(np.zeros((33, 4)))
            return None
    
    def preprocess_sequence(self):
        """
        Temporal sampling -> re-ordering & padding -> normalization"""
        if len(self.skeleton_buffer) < FRAMES_LEN:
            return None # Not enough frames yet
        
        # Get data (already size 64 due to deque maxlen)
        data = np.array(self.skeleton_buffer) # (FRAMES_LEN, 33, 4)
        
        # Skeleton Re-ordering & Padding (33 -> 36)
        T, V_old, C = data.shape
        new_data = np.zeros((T, 36, 3)) # Take only XYZ
        for i, joint_idx in enumerate(self.new_order):
            if joint_idx != -1:
                new_data[:, i, :] = data[:, joint_idx, :3]

        # Normalize (Central to Hip Center)
        left_hip = data[:, 23, :3]
        right_hip = data[:, 24, :3]
        center = (left_hip + right_hip) / 2.0
        new_data = new_data - center[:, None, :] # (FRAMES_LEN, 36, 3)

        # Scale Normalization
        left_shoulder = new_data[:, 0, :]
        right_shoulder = new_data[:, 4, :]

        torso_size = np.linalg.norm(left_shoulder - right_shoulder, axis=1).mean()
        if torso_size > 1e-6:
            new_data = new_data / torso_size

        # Permute to Model Input Format (Batch=1, Channels=3, Frames=64, Joints=36, Person=1)
        tensor = torch.FloatTensor(new_data).permute(2, 0, 1).unsqueeze(-1).unsqueeze(0) # (1, 3, T, 36, 1)
        return tensor
    
def main():
    # Load model
    model = create_mediapipe_skateformer(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    model.eval()

    # Setup video
    cap = cv.VideoCapture(0)
    processor = RealTimeProcessor()

    # State variables for display
    current_pred = "Waiting for data..."
    current_conf = 0.0
    frame_count = 0
    inference_time = 0.0

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    print("Starting real-time inference... Press 'q' to quit.")

    while cap.isOpened():
        sucess, frame = cap.read()
        if not sucess:
            print("End of video stream.")
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process current frame
        viz_object = processor.process_frame(frame_rgb)

        # Only inference if buffer is full FRAMES_LEN
        if len(processor.skeleton_buffer) == FRAMES_LEN and frame_count % 5 == 0:
            input_tensor = processor.preprocess_sequence()
            if input_tensor is not None:
                input_tensor = input_tensor.to(DEVICE)
                index_t = torch.arange(FRAMES_LEN).unsqueeze(0).to(DEVICE)
                
                start_time = time.time()
                with torch.no_grad():
                    output = model(input_tensor, index_t)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    conf, idx = torch.max(probs, dim=1)

                if conf.item() > 0.7:
                    # Update display variables
                    current_conf = conf.item()
                    current_pred = NTU60_CLASSES.get(idx.item(), f"Class {idx.item()}")

                inference_time = (time.time() - start_time)

        if viz_object:
            mp_drawing.draw_landmarks(frame, viz_object, mp_pose.POSE_CONNECTIONS)
            
        ## Draw UI
        cv.rectangle(frame, (0, 0), (640, 100), (245, 117, 16), -1)
        cv.putText(frame, f"Action: {current_pred}", (10, 40), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(frame, f"Conf: {current_conf:.1%}", (10, 70), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(frame, f"Inference: {inference_time:.4f} s", (10, 95), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow("SkateFormer Real-Time", frame)
        frame_count += 1

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()