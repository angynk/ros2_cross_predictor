import json
import rclpy
import yaml
import torch
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
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
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.distance_extractor = DistanceExtractor(device)
                self.get_logger().info(f"Initializing DistanceExtractor on device: {device}")
                image_sub = message_filters.Subscriber(self, Image, '/image_raw')
                detections_sub = message_filters.Subscriber(self, String, '/yolo/detections')
                self.sync = message_filters.ApproximateTimeSynchronizer(
                    [image_sub, detections_sub], queue_size=10, slop=0.15, allow_headerless=True)
                self.sync.registerCallback(self.listener_callback)
                self.get_logger().info('Subscribed to /yolo/image + /yolo/detections')

            elif self.distance_source == "lidar":
                self.get_logger().info("Using distance from LiDAR point cloud")
                self.height_min = settings.get('LIDAR_HEIGHT_MIN', 0.5)
                self.height_max = settings.get('LIDAR_HEIGHT_MAX', 2.0)
                self.subscription = self.create_subscription(
                    PointCloud2, '/velodyne_points', self.lidar_callback, 10)

            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST,
                             depth=10)
            self.publisher = self.create_publisher(Result, '/distance/resultv2', qos)
            
        else:
            self.get_logger().info("Distance feature will not be used (set DISTANCE_SOURCE to 'none')")

    def lidar_callback(self, msg: PointCloud2):
        # Parse PointCloud2: flat float32 layout [x, y, z, intensity] per point
        points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        z = points[:, 2]
        ped_points = points[(z >= self.height_min) & (z <= self.height_max)]
        if ped_points.shape[0] == 0:
            return
        dists = np.sqrt(ped_points[:, 0] ** 2 + ped_points[:, 1] ** 2)
        raw_dist = float(dists.min())
        # Reuse the same labelling logic
        proxy = String()
        proxy.data = str(raw_dist)
        self.calculate_callback(proxy)

    def calculate_callback(self, msg):
        raw_dist = float(msg.data)
        self.get_logger().info(f"Received raw distance: {raw_dist:.2f} m")
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
        self.get_logger().info(f"Published distance result: {label} (raw: {raw_dist})")

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
