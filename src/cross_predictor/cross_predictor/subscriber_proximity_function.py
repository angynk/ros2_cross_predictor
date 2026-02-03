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

from cross_predictor.features_extractor.yolov_detector import YOLOVDetector
from cross_predictor.features_extractor.pose_extractor import PoseExtractor
from cross_predictor.features_extractor.road_context_extractor import RoadContextDetector



class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('proximity_subscriber')

        self.declare_parameter('topic_name', '/image_raw')
        self.topic_name = self.get_parameter('topic_name').value

        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.publisher = self.create_publisher(Image, '/proximity/result', 10)   
        self.bridge = CvBridge()
        self.yolov_detector = YOLOVDetector()
        self.pose_extractor = PoseExtractor() 
        with open('src/cross_predictor/cross_predictor/features_extractor/config.yaml') as f:
            settings = yaml.load(f, Loader=yaml.SafeLoader)
        #DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
        self.road_detector = RoadContextDetector( torch.device('cpu'), settings,(129 * 1.7))
        self.get_logger().info(f'Subscribed to {self.topic_name}')


    def listener_callback(self, msg:Image):
        #self.get_logger().info('I heard: "%s"' % msg.height)
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info('I heard: "%s"' % msg.header.frame_id)
        results = self.yolov_detector.track_pedestrians(msg.header.frame_id)
        proximity_results = []
        for result in results:
            id_person = int(result.boxes.id.numpy()[0])
            img_mod = self.road_detector.prepare_img(result.orig_img)
            self.road_detector.detect_road_context(img_mod,result.orig_img)
            proximity = id_person.__str__() +'-'+ self.road_detector.pedestrian_near_road(result.boxes.xyxy[0].cpu().numpy())
            self.get_logger().info('Proximity: "%s"' % proximity)
            proximity_results.append(proximity)  
        result = Result()
        result.header = msg.header
        result.header.stamp = msg.header.stamp
        result.header.result = proximity_results.__str__()
        self.publisher.publish(result)



def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
