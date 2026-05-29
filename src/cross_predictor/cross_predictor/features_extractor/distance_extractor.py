import os
import torch
from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np
from cross_predictor.features_extractor.yolov_detector import YOLOVDetector

_HF_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

class DistanceExtractor:
    def __init__(self, device, model_dir: str = "models/depth_anything"):
        self.device = device
        self.model = self._load_pipeline(model_dir)
        self.model.model.to(self.device)
        self.model.model.half()
        torch.backends.cudnn.benchmark = True

    def _load_pipeline(self, model_dir: str):
        if os.path.isdir(model_dir):
            return hf_pipeline("depth-estimation", model=model_dir, device=self.device)
        # First run: download from HF Hub and save to model_dir for all future runs
        pipe = hf_pipeline("depth-estimation", model=_HF_MODEL_ID, device=self.device)
        pipe.model.save_pretrained(model_dir)
        processor = getattr(pipe, "image_processor", None) or getattr(pipe, "feature_extractor", None)
        if processor is not None:
            processor.save_pretrained(model_dir)
        return pipe
        

    def get_distance_label(self, depth_value):
        # depth_value range: 0-255, higher = closer (disparity)
        if depth_value >= 160:
            return "TooNearToEgoVeh"
        elif depth_value >= 100:
            return "NearToEgoVeh"
        elif depth_value >= 55:
            return "MiddleDisToEgoVeh"
        elif depth_value >= 25:
            return "FarToEgoVeh"
        else:
            return "TooFarToEgoVeh"
        
    def get_depth(self, image):
        image = Image.fromarray(image)
        depth = np.array(self.model(image)["depth"])
        return depth