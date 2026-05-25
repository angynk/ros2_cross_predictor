import torch
from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np
from cross_predictor.features_extractor.yolov_detector import YOLOVDetector

class DistanceExtractor:
    def __init__(self, device):
        self.device = device
        self.model = hf_pipeline("depth-estimation", 
                                 model="depth-anything/Depth-Anything-V2-Small-hf",device=self.device)
        self.model.model.to(self.device)
        self.model.model.half()
        torch.backends.cudnn.benchmark = True
        

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