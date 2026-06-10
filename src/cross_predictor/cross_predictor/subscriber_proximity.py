import json
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import torch
import yaml
import time

from sensor_msgs.msg import Image
from std_msgs.msg import String
from my_msgs.msg import Result
from cv_bridge import CvBridge
import message_filters

from cross_predictor.features_extractor.road_context_extractor import RoadContextDetector


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('proximity_subscriber')
        self.counter = 0
        self.bridge = CvBridge()

        with open('src/cross_predictor/cross_predictor/config.yaml') as f:
            settings = yaml.load(f, Loader=yaml.SafeLoader)
        self.predictor_type = settings['PREDICTOR']
        self.road_detector = RoadContextDetector(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), settings, (129 * 1.7))

        image_sub = message_filters.Subscriber(self, Image, '/image_raw')
        detections_sub = message_filters.Subscriber(self, String, '/yolo/detections')
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, detections_sub], queue_size=10, slop=0.05, allow_headerless=True)
        self.sync.registerCallback(self.listener_callback)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST,
                         depth=10)
        self.publisher = self.create_publisher(Result, '/proximity', qos_profile=qos)
        self._warm_timer = self.create_timer(0.1, self._keep_gpu_warm)
        self.get_logger().info('Subscribed to /yolo/image + /yolo/detections')

    def _keep_gpu_warm(self):
        self.road_detector.keep_warm()

    def listener_callback(self, img_msg: Image, det_msg: String):
        self.counter += 1
        t0 = time.perf_counter()
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        t1 = time.perf_counter()
        img_mod = self.road_detector.prepare_img(cv_image)
        t2 = time.perf_counter()
        self.road_detector.detect_road_context(img_mod, cv_image)
        t3 = time.perf_counter()

        data = json.loads(det_msg.data)
        proximity_results = {}
        for det in data['detections']:
            xywh = np.array(det['xywh'], dtype=np.float32)
            track_id = det['track_id']

            proximity = str(self.road_detector.pedestrian_near_road(xywh))
            if self.predictor_type == 'KG':
                proximity = self.road_detector.get_proximity(proximity)
            proximity_results[track_id] = proximity

        result = Result()
        result.header = img_msg.header
        result.header.stamp = img_msg.header.stamp
        result.result = proximity_results.__str__()
        t4 = time.perf_counter()
        self.get_logger().info(
            f"[{self.counter}] bridge={t1-t0:.4f}s prep={t2-t1:.4f}s detect={t3-t2:.4f}s rest={t4-t3:.4f}s total={t4-t0:.4f}s"
        )
        #self.get_logger().info(f"Proximity results: {result.result}")
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
