import cv2 as cv
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
import sys
import os

sys.path.insert(0, '/mnt/A87E3F8E7E3F5472/CapstoneProject')

from my_src.utils.mediapipe2ntu_mapping import MediaPipeToNTUConverter
from my_src.models.SkateFormer import SkateFormer_
from my_src.utils.temporal_smoothing import temporal_smoothing, temporal_smoothing_bone_aware

class SkeletonActionRecognizer:
    def __init__(self, model_path: str, num_frames=64, num_points=48, num_people=2, device='cpu'):
        self.device = device
        self.num_frames = num_frames
        self.num_points = num_points
        self.num_people = num_people
        self.converter = MediaPipeToNTUConverter()
        self.ntu_edges = MediaPipeToNTUConverter.get_connections()

        # Load SkateFormer model
        self.model = SkateFormer_(
                    in_channels=3,
                    num_classes=60,
                    num_frames=64,
                    num_points=24,
                    num_people=2,
                    index_t=True,
                    type_1_size=(8, 8),
                    type_2_size=(8, 12),
                    type_3_size=(8, 8),
                    type_4_size=(8, 12),
                    attn_drop=0.5,
                    drop_path=0.2,
                    mlp_ratio=4.0
                )

        weights = torch.load(model_path, map_location=device)
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.model.eval()

        # MediaPipe Pose setup 
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="./my_src/pretrained/pose_landmarks/pose_landmarker_full.task"),
            running_mode=VisionRunningMode.VIDEO,
        )

        self.pose_landmarker = PoseLandmarker.create_from_options(options)

        # Frame buffer for temporal sequence
        self.skeleton_buffer = []
        self.frame_count = 0

        # Action classes
        self.action_classes = self._get_ntu60_classes()

    def _get_ntu60_classes(self):
        classes = [
            'drink water', 'eat meal/snack', 'brush teeth', 'brush hair', 'drop',
            'pick up', 'throw', 'sit down', 'stand up', 'clapping',
            'reading', 'writing', 'tear up paper', 'wear jacket', 'take off jacket',
            'wear a shoe', 'take off a shoe', 'wear on glasses', 'take off glasses',
            'put on a hat/cap', 'take off a hat/cap', 'cheer up', 'hand waving',
            'kicking something', 'reach into pocket', 'hopping', 'jump up',
            'make a phone call/answer phone', 'play with phone/tablet',
            'type on a keyboard', 'point to something with finger', 'taking a selfie',
            'check time (from watch)', 'rub two hands together', 'nod head/bow',
            'shake head', 'wipe face', 'salute', 'put palms together',
            'cross hands in front', 'sneeze/cough', 'staggering', 'falling down',
            'headache', 'chest pain', 'back pain', 'neck pain', 'nausea/vomiting',
            'fan self', 'punch/slap other person', 'kick other person',
            'push other person', 'pat on back of other person',
            'point finger at the other person', 'hugging other person',
            'giving something to other person', 'touch other person pocket',
            'shaking hands', 'walking towards each other',
            'walking apart from each other'
        ]

        return classes
    
    def process_frame(self, frame):

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect pose
        detection_result = self.pose_landmarker.detect_for_video(mp_image, self.frame_count)
        self.frame_count += 1

        if not detection_result.pose_world_landmarks:
            return None

        # Convert to NTU format
        landmarks = detection_result.pose_world_landmarks[0]
        ntu_skeleton = self.converter.convert(landmarks)

        return ntu_skeleton
    
    def add_skeleton(self, skeleton):
        """Add detected skeleton to frame buffer"""
        if skeleton is not None:
            self.skeleton_buffer.append(skeleton)

            if len(self.skeleton_buffer) > self.num_frames:
                self.skeleton_buffer.pop(0)

    def is_ready(self):
        """Check if buffer has enough frames"""
        return len(self.skeleton_buffer) == self.num_frames
    
    def predict(self, apply_smoothing=True, smoothing_alpha=0.8):
        if not self.is_ready():
            return None, None, None
        
        # Prepare input tensor: (T, V, C) -> (C, T, V, M) -> (1, C, T, V, M)
        skeleton_seq = np.array(self.skeleton_buffer) # Shape: (64, 24, 3)

        if apply_smoothing:
            skeleton_seq = temporal_smoothing(
                skeleton_seq,
                alpha=smoothing_alpha
            )

        # !!! 2 people (test for 1 person input) !!!
        skeleton_p1 = skeleton_seq
        skeleton_p2 = np.zeros_like(skeleton_seq)
        skeleton_both = np.stack([skeleton_p1, skeleton_p2], axis=-1)

        # Transpose
        input_data = skeleton_both.transpose(2, 0, 1, 3)
        
        # Add batch dimension
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Temporal index
        index_t = torch.arange(self.num_frames, dtype=torch.long).unsqueeze(0).to(self.device) # (1, 64)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor, index_t)
            probabilities = torch.nn.functional.softmax(outputs, 1)
            confidence, predicted = torch.max(probabilities, 1)

        action = self.action_classes[predicted.item()]
        confidence_score = confidence.item()

        # Get top-5 predictions
        top5_prob, top5_idx = torch.topk(probabilities[0], 5)
        top_k_actions = [
            (self.action_classes[idx.item()], prob.item())
            for idx, prob in zip(top5_idx, top5_prob)
        ]
        return action, confidence_score, top_k_actions


if __name__ == "__main__":
    model_path = "my_src/pretrained/NTU60_CView/SkateFormer_j.pt"
    recognizer = SkeletonActionRecognizer(model_path=model_path,
                                          num_frames=64,
                                          num_points=24,
                                          num_people=2,
                                          device='cpu')

    video_path = "/mnt/A87E3F8E7E3F5472/CapstoneProject/my_src/video/S001C001P001R001A001_rgb.avi"
    cap = cv.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        skeleton = recognizer.process_frame(frame)
        if skeleton is not None:
            recognizer.add_skeleton(skeleton)

        if recognizer.is_ready():
            action_name, confidence, top_k = recognizer.predict()
            if action_name is not None:
                print(f"Predicted Action: {action_name} (Confidence: {confidence:.4f})")
                print("Top 5 predictions:")
                for i, (action, prob) in enumerate(top_k, 1):
                    print(f"  {i}. {action}: {prob:.4f}")
                print("=" * 60)
            cv.putText(frame, f"Action: {action_name} ({confidence:.2f})", (30, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("Skeleton Action Recognition", cv.resize(frame, (800, 600)))

        if cv.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()