import numpy as np

def temporal_smoothing(skel_seq, alpha=0.8):
    """
    Exponential Moving Average smoothing for skeleton sequences.
    
    Args: 
        skel_seq : np.ndarray (T, V, C)
        alpha: Control the smoothing
    
    Returns:
        smoothed_ske_seq : np.ndarray (T, V, C)
    """

    T, V, C = skel_seq.shape
    smoothed = np.zeros_like(skel_seq)

    smoothed[0] = skel_seq[0]

    for t in range(1, T):
        smoothed[t] = alpha * smoothed[t-1] + (1 - alpha) * skel_seq[t]

    return smoothed


def temporal_smoothing_bone_aware(
    skel_seq,
    edges,
    alpha=0.8
):
    """
    EMA smoothing with bone-length correction
    
    Args:
        skel_seq: (T, V, C)
        edges: list of (parent, child) joint indices
    """
    smoothed = temporal_smoothing(skel_seq, alpha)

    # enforce bone length consistency
    for t in range(1, smoothed.shape[0]):
        for i, j in edges:
            vec = smoothed[t, j] - smoothed[t, i]
            prev_len = np.linalg.norm(
                skel_seq[t - 1, j] - skel_seq[t - 1, i]
            ) + 1e-6
            cur_len = np.linalg.norm(vec) + 1e-6
            smoothed[t, j] = smoothed[t, i] + vec * (prev_len / cur_len)

    return smoothed

