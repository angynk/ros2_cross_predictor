# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cv2
import rclpy
from rclpy.node import Node
import torch
import yaml
from pathlib import Path



from sensor_msgs.msg import Image
from my_msgs.msg import Result
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cross_predictor.features_extractor.yolov_detector import YOLOVDetector
from cross_predictor.features_extractor.pose_extractor import PoseExtractor
from cross_predictor.features_extractor.action_extractor import ActionRecognizer

import sys
print(sys.path)
print(sys.executable)

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('attention_subscriber')

        self.declare_parameter('topic_name', '/image_raw')
        self.topic_name = self.get_parameter('topic_name').value

        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, # Don't wait for retries
                        history=HistoryPolicy.KEEP_LAST,          # Only keep the newest
                        depth=10)
        self.publisher = self.create_publisher(Result, '/action/resultv2', qos)    
        self.bridge = CvBridge()
        self.yolov_detector = YOLOVDetector()
        self.pose_extractor = PoseExtractor() 
        with open('src/cross_predictor/cross_predictor/config.yaml') as f:
            settings = yaml.load(f, Loader=yaml.SafeLoader)
        self.predictor_type = settings['PREDICTOR']
        #DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
        self.action_recognizer = ActionRecognizer(settings, torch.device('cpu'))
        self.get_logger().info(f'Subscribed to {self.topic_name}')


    def listener_callback(self, msg:Image):
        #self.get_logger().info('I heard: "%s"' % msg.height)
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        #self.get_logger().info('I heard: "%s"' % msg.header.frame_id)
        results = self.yolov_detector.track_pedestrians(cv_image)
        #self.get_logger().info(f"Detected {len(results)} pedestrians in frame {msg.header.frame_id}")
        action_results = {}
        for result in results:
            if result.boxes.id is not None:
                ids = result.boxes.id.numpy()
                for i in range(len(ids)):
                    boxes = result.boxes.xywh[i].cpu().numpy()
                    id_person_bbox = self.yolov_detector.id_from_bbox(boxes)
                    id_person = int(result.boxes.id.numpy()[i])
                    _,_, skeleton = self.pose_extractor.extract_pose(result.orig_img, result.boxes.xywh[i].cpu().numpy())
                    buffer = self.action_recognizer.save_buffer_skeleton(id_person, skeleton)
                    action = str(self.action_recognizer.detect_action(skeleton, buffer))
                    if self.predictor_type=='KG':
                        action = self.action_recognizer.get_action(action)
                    action_results[id_person_bbox] = action

        result = Result()
        result.header = msg.header
        result.header.stamp = msg.header.stamp
        result.result = action_results.__str__()
        #self.get_logger().info(f"Publishing Action Results: {result.result} for frame {msg.header.frame_id}")
        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
