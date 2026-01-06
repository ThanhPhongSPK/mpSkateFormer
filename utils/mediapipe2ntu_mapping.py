import numpy as np

def rotate_z_90(skeleton):
    rotated_skeleton = skeleton.copy()
    rotated_skeleton[:, 0] = -skeleton[:, 1]  # X_new = -Y_old
    rotated_skeleton[:, 1] = skeleton[:, 0]   # Y_new = X_old
    rotated_skeleton[:, 2] = skeleton[:, 2]   # Z_new = Z_old (unchanged)
    return rotated_skeleton

class MediaPipeToNTUConverter:
    """Converts MediaPipe pose landmarks to NTU skeleton format"""
    
    # MediaPipe Pose indexes
    MP_NOSE = 0
    MP_LEFT_SHOULDER = 11
    MP_RIGHT_SHOULDER = 12
    MP_LEFT_ELBOW = 13
    MP_RIGHT_ELBOW = 14
    MP_LEFT_WRIST = 15
    MP_RIGHT_WRIST = 16
    MP_LEFT_INDEX_TIP = 19
    MP_RIGHT_INDEX_TIP = 20
    MP_LEFT_THUMB = 21
    MP_RIGHT_THUMB = 22
    MP_LEFT_HIP = 23
    MP_RIGHT_HIP = 24
    MP_LEFT_KNEE = 25
    MP_RIGHT_KNEE = 26
    MP_LEFT_ANKLE = 27
    MP_RIGHT_ANKLE = 28
    MP_LEFT_FOOT_TIP = 31
    MP_RIGHT_FOOT_TIP = 32

    NTU_CONNECTIONS_24 = [
        # Spine chain
        (0, 1), (1, 2), (2, 3), 
        # Left arm
        (2, 4), (4, 5), (5, 6), (6, 7), (7, 20), (7, 21),
        # Right arm
        (2, 8), (8, 9), (9, 10), (10, 11), (11, 22), (11, 23),
        # Left leg
        (0, 12), (12, 13), (13, 14), (14, 15),
        # Right leg
        (0, 16), (16, 17), (17, 18), (18, 19)
]
    
    def __init__(self):
        """Initialize the converter"""
        pass
    
    def _to_array(self, landmark):
        """Convert MediaPipe landmark to numpy array"""
        return np.array([landmark.x, landmark.y, landmark.z])
    
    def convert(self, mp_landmarks):

        # Define the ntu matrix
        ntu_skeleton = np.zeros((25, 3))
        
        # Get all MediaPipe landmarks
        left_hip = self._to_array(mp_landmarks[self.MP_LEFT_HIP])
        right_hip = self._to_array(mp_landmarks[self.MP_RIGHT_HIP])
        left_shoulder = self._to_array(mp_landmarks[self.MP_LEFT_SHOULDER])
        right_shoulder = self._to_array(mp_landmarks[self.MP_RIGHT_SHOULDER])
        nose = self._to_array(mp_landmarks[self.MP_NOSE])
        left_wrist = self._to_array(mp_landmarks[self.MP_LEFT_WRIST])
        right_wrist = self._to_array(mp_landmarks[self.MP_RIGHT_WRIST])
        left_index_tip = self._to_array(mp_landmarks[self.MP_LEFT_INDEX_TIP])
        right_index_tip = self._to_array(mp_landmarks[self.MP_RIGHT_INDEX_TIP])

        # ---- Interpolate missing NTU joints ----
        # (1) base of spine
        ntu_skeleton[0] = (left_hip + right_hip) / 2

        # (21) spine
        ntu_skeleton[20] = (left_shoulder + right_shoulder) / 2
        
        # (2) middle of spine
        ntu_skeleton[1] = (ntu_skeleton[0] + ntu_skeleton[20]) / 2

        # (4) head
        ntu_skeleton[3] = nose

        # (3) neck: Middle point between spine and head
        ntu_skeleton[2] = (ntu_skeleton[20] + ntu_skeleton[3]) / 2

        # (8) left hand
        ntu_skeleton[21] = left_index_tip
        ntu_skeleton[6] = left_wrist
        ntu_skeleton[7] = (ntu_skeleton[6] + ntu_skeleton[21]) / 2

        # (12) right hand
        ntu_skeleton[23] = right_index_tip
        ntu_skeleton[10] = right_wrist
        ntu_skeleton[11] = (ntu_skeleton[10] + ntu_skeleton[23]) / 2

        # ---- Directly mapping NTU joints ----
        ntu_skeleton[4] = left_shoulder
        ntu_skeleton[5] = self._to_array(mp_landmarks[self.MP_LEFT_ELBOW])
        ntu_skeleton[8] = right_shoulder
        ntu_skeleton[9] = self._to_array(mp_landmarks[self.MP_RIGHT_ELBOW])
        ntu_skeleton[12] = left_hip
        ntu_skeleton[13] = self._to_array(mp_landmarks[self.MP_LEFT_KNEE])
        ntu_skeleton[14] = self._to_array(mp_landmarks[self.MP_LEFT_ANKLE])
        ntu_skeleton[15] = self._to_array(mp_landmarks[self.MP_LEFT_FOOT_TIP])
        ntu_skeleton[16] = right_hip
        ntu_skeleton[17] = self._to_array(mp_landmarks[self.MP_RIGHT_KNEE])
        ntu_skeleton[18] = self._to_array(mp_landmarks[self.MP_RIGHT_ANKLE])
        ntu_skeleton[19] = self._to_array(mp_landmarks[self.MP_RIGHT_FOOT_TIP])
        ntu_skeleton[22] = self._to_array(mp_landmarks[self.MP_LEFT_THUMB])
        ntu_skeleton[24] = self._to_array(mp_landmarks[self.MP_RIGHT_THUMB])
        
        # --- Transform to NTU coordinate system ---
        # Re-center
        origin = ntu_skeleton[0].copy()
        ntu_skeleton -= origin

        # # Invert Z axis
        # ntu_skeleton[:, 2] *= -1

        # # Invert Y axis
        # ntu_skeleton[:, 1] *= -1

        x = ntu_skeleton[:, 0]
        y = ntu_skeleton[:, 2]   # forward
        z = -ntu_skeleton[:, 1]  # up (invert)

        ntu_skeleton[:, 0] = x
        ntu_skeleton[:, 1] = y
        ntu_skeleton[:, 2] = z

        excluded_joints = [20]
        ntu_skeleton = np.delete(ntu_skeleton, excluded_joints, axis=0)

        return ntu_skeleton


    @staticmethod
    def get_connections():
        """Get NTU skeleton connection pairs"""
        return MediaPipeToNTUConverter.NTU_CONNECTIONS_24