"""
One-time conversion: YOLOPv2 TorchScript → TensorRT engine (saved as TorchScript).

Run once before launching the ROS node:
    python3 convert_yolopv2_trt.py

The output model (yolopv2_trt.ts) can be loaded with torch.jit.load() exactly
like the original, so inference code needs only a path change.
"""
import torch
import torch_tensorrt
import yaml
from pathlib import Path

CONFIG = '/home/angie-melo/Documents/PHD/ros2_cross_predictor/src/cross_predictor/cross_predictor/config.yaml'

with open(CONFIG) as f:
    settings = yaml.load(f, Loader=yaml.SafeLoader)

src_path = settings['F_YOLOPV2']
trt_path = str(Path(src_path).with_suffix('')) + '_trt.ts'

device = torch.device('cuda:0')

print(f'Loading {src_path} ...')
model = torch.jit.load(src_path, map_location=device)
model = model.half().eval()

example = torch.zeros(1, 3, 640, 640, device=device, dtype=torch.half)

print('Compiling with TensorRT (this takes a few minutes the first time) ...')
model_trt = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        min_shape=(1, 3, 640, 640),
        opt_shape=(1, 3, 640, 640),
        max_shape=(1, 3, 640, 640),
        dtype=torch.half,
    )],
    enabled_precisions={torch.half},
    truncate_long_and_double=True,
)

torch.jit.save(model_trt, trt_path)
print(f'Saved TRT model to {trt_path}')
print('Add this to config.yaml:')
print(f"  F_YOLOPV2_TRT: '{trt_path}'")
