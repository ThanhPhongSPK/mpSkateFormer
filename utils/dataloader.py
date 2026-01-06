import torch
import os
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

NTU60_CLASSES = {
            0: "drink water", 1: "eat meal/snack", 2: "brushing teeth", 3: "brushing hair",
            4: "drop", 5: "pickup", 6: "throw", 7: "sit down", 8: "stand up", 
            9: "clapping", 10: "reading", 11: "writing", 12: "tear up paper", 
            13: "put on jacket", 14: "take off jacket", 15: "put on a shoe", 
            16: "take off a shoe", 17: "put on glasses", 18: "take off glasses", 
            19: "put on a hat/cap", 20: "take off a hat/cap", 21: "cheer up", 
            22: "hand waving", 23: "kicking something", 24: "reach into pocket", 
            25: "hopping", 26: "jump up", 27: "phone call", 28: "play with phone/tablet", 
            29: "type on a keyboard", 30: "point to something", 31: "taking a selfie", 
            32: "check time (from watch)", 33: "rub two hands", 34: "nod head/bow", 
            35: "shake head", 36: "wipe face", 37: "salute", 38: "put palms together", 
            39: "cross hands in front", 40: "sneeze/cough", 41: "staggering", 
            42: "falling", 43: "headache", 44: "chest pain", 45: "back pain", 
            46: "neck pain", 47: "nausea/vomiting", 48: "fan self", 
            49: "punch/slap", 50: "kicking other person", 51: "pushing other person", 
            52: "pat on back", 53: "point finger at other", 54: "hugging other person", 
            55: "giving object to other", 56: "touch other person's pocket", 
            57: "shaking hands", 58: "walking towards each other", 
            59: "walking apart from each other"
        }

def _get_ntu60_split(base_folders_list, split_type='xsub'):
    """
    Split train and val base one Subject ID
    """
    train_files = []
    val_files = []



    train_subjects = [
        1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
    ]

    for folder_path in base_folders_list:
        if not os.path.exists(folder_path):
            continue
        
        for f in os.listdir(folder_path):
            if not f.endswith('.npy'):
                continue
            
            try:
                # Find sequence with 'P'
                part_p = f.split('P')[1]
                subject_id = int(part_p[:3])

                full_path = os.path.join(folder_path, f)

                if split_type == 'xsub':
                    if subject_id in train_subjects:
                        train_files.append(full_path)
                    else:
                        val_files.append(full_path)

            except IndexError:
                print(f"Skipping invalid filename: {f}")
    
    return train_files, val_files


class MediaPipeSkateDataset(Dataset):
    def __init__(self, file_list, frames_len=64, mode='train'):
        self.files = file_list
        self.frames_len = frames_len
        self.mode = mode

        # Padding from 33 to Raumania 36
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

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        raw_data = np.load(file_path) # T x 33 x 4

        # Sampling (Resize time dimension)
        data = self.temporal_sampling(raw_data, self.frames_len)

        # Skeleton Re-ordering & Padding (33 -> 36)
        T, V_old, C = data.shape
        new_data = np.zeros((T, 36, 3)) # Take only XYZ
        for i, joint_idx in enumerate(self.new_order):
            if joint_idx == -1:
                new_data[:, i, :] = 0.0 # Padding
            else:
                new_data[:, i, :] = data[:, joint_idx, :3]

        # Normalize (Central to Hip Center)
        left_hip = data[:, 23, :3]
        right_hip = data[:, 24, :3]
        center = (left_hip + right_hip) / 2.0
        new_data = new_data - center[:, None, :] # (T, 36, 3)

        # Scale Normalization
        left_shoulder = new_data[:, 0, :]
        right_shoulder = new_data[:, 4, :]

        torso_size = np.linalg.norm(left_shoulder - right_shoulder, axis=1).mean()
        if torso_size > 1e-6:
            new_data = new_data / torso_size
        else:
            pass # Avoid division by zero

        # Data Augmentation (Train mode only)
        if self.mode == 'train':
            new_data = self.augment(new_data)

        # Final input (C, T, V, M)
        # SkateFormer input: (Batch, Channels, Frames, Joints, Persons)
        out_tensor = torch.FloatTensor(new_data).permute(2, 0, 1).unsqueeze(-1) # (3, T, 36, 1)
        label = self.get_label_from_name(os.path.basename(file_path))

        # Skate-Embedding
        index_t = torch.arange(self.frames_len)

        return out_tensor, label, index_t

    def temporal_sampling(self, data, target_len):
        # Keep the logic sampling uniformly
        original_len = data.shape[0]
        if original_len == 0: return np.zeros((target_len, 33, 4))
        
        if self.mode == 'train':
            indices = []
            if original_len < target_len:
                indices = np.sort(np.random.choice(original_len, target_len, replace=True))
            else:
                # Split equally into bins and take randomly in each bin
                interval = original_len / target_len
                for i in range(target_len):
                    idx = int(np.random.uniform(i * interval, (i+1) * interval))
                    idx = min(idx, original_len-1)
                    indices.append(idx)
            indices = np.array(indices)
        else:
            indices = np.linspace(0, original_len-1, target_len).astype(int)
        
        sampled_data = data[indices]
        return sampled_data
        
    def augment(self, data):
        if random.random() < 0.5:
            # Rotation around Y-axis (vertical)
            theta = random.uniform(-0.3, 0.3) # +/- -17 degrees
            c, s = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            data = np.dot(data, rotation_matrix)

        if random.random() < 0.5:
            # Scaling
            scale = random.uniform(0.85, 1.15)
            data = data * scale

        if random.random() < 0.5:
            # Gaussian Noise
            noise = np.random.normal(0, 0.02, data.shape)
            data = data + noise
        return data
    
    def get_label_from_name(self, filename):
        try:
            return int(filename.split('A')[1][:3]) - 1
        except:
            return 0

    