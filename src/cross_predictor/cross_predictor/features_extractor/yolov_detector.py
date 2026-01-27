from ultralytics import YOLO

class YOLOVDetector:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def detect_pedestrians(self, image):
        results = self.model.predict(image, classes=[0])
        return results
    
    def track_pedestrians(self,image):
        results = self.model.track(image, classes=[0])
        return results