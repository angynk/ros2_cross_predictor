import torch
import yaml
import cv2
import hashlib
from pathlib import Path
from yaml.loader import SafeLoader
from cross_predictor.features_extractor.yolov_detector import YOLOVDetector
from cross_predictor.features_extractor.pose_extractor import PoseExtractor
from cross_predictor.features_extractor.action_extractor import ActionRecognizer
from cross_predictor.features_extractor.road_context_extractor import RoadContextDetector
from cross_predictor.features_extractor.attention_extractor import pedestrian_gaze

from cross_predictor.kge.kg_predictor import KGPredictor

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def extract_orientation(image_path):
    yolov_detector = YOLOVDetector()
    pose_extractor = PoseExtractor()
    results = yolov_detector.detect_pedestrians(image_path)
    for result in results:
        orientation, skeleton = pose_extractor.extract_pose(result.orig_img, result.boxes.xywh[0].cpu().numpy())
        print("Orientation: ", orientation)

    
def extract_attention(image_path):
    yolov_detector = YOLOVDetector()
    pose_extractor = PoseExtractor()
    results = yolov_detector.detect_pedestrians(image_path)
    for result in results:
        orientation, skeleton = pose_extractor.extract_pose(result.orig_img, result.boxes.xywh[0].cpu().numpy())
        attention = str(pedestrian_gaze (skeleton))
        print("Attention: ", attention)

def init_action_extractor(settings):
    yolov_detector = YOLOVDetector()
    pose_extractor = PoseExtractor()
    action_recognizer = ActionRecognizer(settings, torch.device('cpu'))
    return yolov_detector, pose_extractor, action_recognizer

def id_from_bbox(bbox):
    box_string = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
    unique_id = hashlib.md5(box_string.encode()).hexdigest()[:8]
    return unique_id

def extract_action(image_path, yolov_detector, pose_extractor, action_recognizer):
    
    results = yolov_detector.track_pedestrians(image_path)
    
    for result in results:

        annotated_frame = result.plot()
        if result.boxes.id is not None:
            ids = result.boxes.id.numpy()
            for i in range(len(ids)):
                boxes = result.boxes.xywh[i].cpu().numpy()
                id_person_bbox = id_from_bbox(boxes)
                print("ID Person: ", id_person_bbox)
                id_person = int(ids[i])
                _, orientation, skeleton = pose_extractor.extract_pose(result.orig_img, boxes)
                buffer = action_recognizer.save_buffer_skeleton(id_person, skeleton)
                action = action_recognizer.detect_action(skeleton, buffer)
                action = action_recognizer.get_action(action)
                print("Action: ", action)
        cv2.imshow("YOLO Tracking", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(0):
            break
        

def init_proximity_extractor(settings, focal_length):
    road_detector = RoadContextDetector(DEVICE, settings,focal_length)
    yolov_detector = YOLOVDetector()
    return road_detector, yolov_detector

def extract_proximity(image_path, road_detector, yolov_detector):
    results = yolov_detector.track_pedestrians(image_path)
    for result in results:
        img_mod = road_detector.prepare_img(result.orig_img)
        road_detector.detect_road_context(img_mod,result.orig_img)
        proximity = road_detector.pedestrian_near_road(result.boxes.xyxy[0].cpu().numpy())
        print("Proximity: ", proximity)


def predict_crossing(settings):
    predictor_kg = KGPredictor(settings)
    frame_features = {'proximity': "NearFromCurb", 'action': "Na", 'distance': "MiddleDisToEgoVeh",
                              'attention': "Looking", 'orientation': "LeftDirection"}
    prediction, prob_cross, prob_nocross = predictor_kg.bayesian_method(frame_features)
    print("Prediction: ", prediction)
    return prediction


with open('src/cross_predictor/cross_predictor/config.yaml') as f:
    settings = yaml.load(f, Loader=SafeLoader)
#extract_orientation("/home/angie-melo/Documents/DataSets/JAAD/images/video_0190/00058.png")
#extract_attention("/home/angie-melo/Documents/DataSets/JAAD/images/video_0190/00058.png")
folder_path = Path('/home/angie-melo/Documents/DataSets/Markus/Participant_02/test')
image_paths = sorted(folder_path.glob('*.jpg'))
'''road_detector, yolov_detector= init_proximity_extractor(settings, (129 * 1.7))
for path in image_paths:
    extract_proximity(path,road_detector, yolov_detector)'''
yolov_detector, pose_extractor, action_recognizer = init_action_extractor(settings)
#for path in image_paths:
#    extract_action(path, yolov_detector, pose_extractor, action_recognizer)
#predict_crossing(settings)


