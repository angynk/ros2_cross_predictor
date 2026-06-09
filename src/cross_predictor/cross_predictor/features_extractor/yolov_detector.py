import yaml
from ultralytics import YOLO
import hashlib
import uuid

ID_EXPIRY_FRAMES = 120

class YOLOVDetector:
    def __init__(self, settings=None):
        import torch
        if settings is None:
            with open('src/cross_predictor/cross_predictor/config.yaml') as f:
                settings = yaml.safe_load(f)
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(settings['YOLO_WEIGHTS'])
        # tracker_id -> unique_id
        self._id_map: dict[int, str] = {}
        # tracker_id -> last frame it was seen
        self._last_seen: dict[int, int] = {}
        # all unique IDs ever assigned (survives ID expiry)
        self._all_ids: set[str] = set()

    def detect_pedestrians(self, image):
        results = self.model.predict(image, classes=[0], device=self.device)
        return results

    def track_pedestrians(self, image):
        results = self.model.track(image, classes=[0], conf=0.40,
                                    iou=0.7, show=False, device=self.device)
        return results

    def resolve_tracking_id(self, tracker_id: int, frame_number: int) -> str:
        """Return a stable unique ID for a YOLO tracker_id.

        If the tracker_id was absent for more than ID_EXPIRY_FRAMES frames,
        it is treated as a new pedestrian and assigned a fresh unique ID.
        """
        last = self._last_seen.get(tracker_id)
        if last is None or (frame_number - last) > ID_EXPIRY_FRAMES:
            self._id_map[tracker_id] = uuid.uuid4().hex[:8]
            self._all_ids.add(self._id_map[tracker_id])
        self._last_seen[tracker_id] = frame_number
        return self._id_map[tracker_id]

    @property
    def pedestrian_count(self) -> int:
        """Total number of unique pedestrians identified since tracking started."""
        return len(self._all_ids)

    def id_from_bbox(self, bbox):
        box_string = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
        unique_id = hashlib.md5(box_string.encode()).hexdigest()[:8]
        return unique_id