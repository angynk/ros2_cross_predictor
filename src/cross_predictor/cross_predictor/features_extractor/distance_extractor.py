from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np
from cross_predictor.features_extractor.yolov_detector import YOLOVDetector

class DistanceExtractor:
    def __init__(self):
        self.model = hf_pipeline("depth-estimation", 
                                 model="depth-anything/Depth-Anything-V2-Small-hf")
        self.yolov_detector = YOLOVDetector()

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
    
    def estimate_distance(self,image):
        #image = Image.open(image_path)
        results = self.yolov_detector.track_pedestrians(image)
        distance_results = {}
        image = Image.fromarray(image)
        depth = np.array(self.model(image)["depth"])
        for result in results:
            if result.boxes.id is not None:
                ids = result.boxes.id.numpy()
                for i in range(len(ids)):
                    id_person_bbox = self.yolov_detector.id_from_bbox(result.boxes.xywh[i].cpu().numpy())
                    boxes = result.boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, boxes)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Sample depth in the pedestrian region (feet area is more reliable)
                    foot_y = min(y2, depth.shape[0] - 1)
                    depth_value = depth[foot_y, cx]
                    distance_results[id_person_bbox] = [depth_value, self.get_distance_label(depth_value)]
        
        
        return distance_results