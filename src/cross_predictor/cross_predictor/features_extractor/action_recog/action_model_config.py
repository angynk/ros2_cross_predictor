from dataclasses import dataclass
import torch.nn as nn
from .action_transformer import Transformer


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float
    n_encoder: int
    embed_dim: int
    n_frames: int
    n_keypoints: int
    mlp_size: int
    device: str


def make_model(config, DEVICE, num_classes):
    model = Transformer(config, num_classes).to(DEVICE)
    # initialize model parameters
    # it seems that this initialization is very important!
    for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model