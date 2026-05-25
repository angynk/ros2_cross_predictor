from ultralytics import YOLO
import hashlib

class YOLOVDetector:
    def __init__(self):
        import torch
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolo26n.pt")

    def detect_pedestrians(self, image):
        results = self.model.predict(image, classes=[0], device=self.device)
        return results

    def track_pedestrians(self,image):
        results = self.model.track(image, classes=[0], conf=0.40,
                                    iou=0.7, show=False, device=self.device)
        return results
    
    def id_from_bbox(self, bbox):
        box_string = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
        unique_id = hashlib.md5(box_string.encode()).hexdigest()[:8]
        return unique_id