import torch
import torch.nn as nn
import onnx 
from my_src.models.SkateFormer import create_mediapipe_skateformer

# 
MODEL_PATH = "___.pt" # File model đã train
ONNX_PATH = "___.onnx"
NUM_CLASSES = 60
FRAMES = 64
JOINTS = 36

def export_to_onnx():
    print(f"Loading trained model from {MODEL_PATH}...")

    device = torch.device('cpu')
    model = create_mediapipe_skateformer(num_classes=NUM_CLASSES)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, FRAMES, JOINTS, 1, device=device)

    # Input 2: Time index
    dummy_index_t = torch.arange(FRAMES, device=device).unsqueeze(0).long()

    print("Exporting to ONNX..")

    torch.onnx.export(
        model,
        (dummy_input, dummy_index_t),
        ONNX_PATH,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names = ['input_skeleton', 'input_time_index'],
        output_names = ['output_logits'],
        dynamic_axes={
            'input_skeleton': {0: 'batch_size'},
            'input_time_index': {0: 'batch_size'},
            'output_logits': {0: 'batch_size'}
        }
    )

    print(f"--> ONNX model saved to: {ONNX_PATH}")

if __name__ == "__main__":
    export_to_onnx()