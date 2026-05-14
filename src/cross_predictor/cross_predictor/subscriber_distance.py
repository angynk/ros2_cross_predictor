import json
import rclpy
import yaml
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from my_msgs.msg import Result
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters

from cross_predictor.features_extractor.distance_extractor import DistanceExtractor


class DistanceCalculator(Node):

    def __init__(self):
        super().__init__('distance_calculator')
        with open('src/cross_predictor/cross_predictor/config.yaml') as f:
            settings = yaml.load(f, Loader=yaml.SafeLoader)
        self.distance_source = settings['DISTANCE_SOURCE']
        if self.distance_source != "none":
            self.predictor_type = settings['PREDICTOR']
            if self.distance_source == "estimation":
                self.get_logger().info("Using distance estimation from image")
                self.bridge = CvBridge()
                self.distance_extractor = DistanceExtractor()

                image_sub = message_filters.Subscriber(self, Image, '/yolo/image')
                detections_sub = message_filters.Subscriber(self, String, '/yolo/detections')
                self.sync = message_filters.ApproximateTimeSynchronizer(
                    [image_sub, detections_sub], queue_size=10, slop=0.05, allow_headerless=True)
                self.sync.registerCallback(self.listener_callback)

            elif self.distance_source == "lidar":
                self.get_logger().info("Using distance from LiDAR")
                self.declare_parameter('topic_name', '/radar/raw_data')
                self.topic_name = self.get_parameter('topic_name').value
                self.subscription = self.create_subscription(
                    String, self.topic_name, self.calculate_callback, 10)

            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST,
                             depth=10)
            self.publisher = self.create_publisher(Result, '/distance/resultv2', qos)
            self.get_logger().info('Subscribed to /yolo/image + /yolo/detections')
        else:
            self.get_logger().info("Distance feature will not be used (set DISTANCE_SOURCE to 'none')")

    def calculate_callback(self, msg):
        raw_dist = float(msg.data)
        label = str(raw_dist)
        if self.predictor_type == 'KG':
            if 0 <= raw_dist <= 2.5:
                label = 'TooNearToEgoVeh'
            elif 2.5 < raw_dist <= 8.0:
                label = 'NearToEgoVeh'
            elif 8.0 < raw_dist <= 20.0:
                label = 'MiddleDisToEgoVeh'
            elif 20.0 < raw_dist <= 45.0:
                label = 'FarToEgoVeh'
            else:
                label = 'TooFarToEgoVeh'
        result = Result()
        result.result = label
        self.publisher.publish(result)

    def listener_callback(self, img_msg: Image, det_msg: String):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        depth = self.distance_extractor.get_depth(cv_image)

        data = json.loads(det_msg.data)
        distance_results = {}
        for det in data['detections']:
            xywh = det['xywh']
            xyxy = det['xyxy']
            track_id = det['track_id']

            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2
            foot_y = min(y2, depth.shape[0] - 1)
            depth_value = depth[foot_y, cx]

            distance_results[track_id] = [
                depth_value,
                self.distance_extractor.get_distance_label(depth_value)
            ]

        result = Result()
        result.header = img_msg.header
        result.header.stamp = img_msg.header.stamp
        result.result = distance_results.__str__()
        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = DistanceCalculator()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
