import json
import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from my_msgs.msg import Result
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters

from cross_predictor.features_extractor.pose_extractor import PoseExtractor
from cross_predictor.features_extractor.attention_extractor import pedestrian_gaze


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('attention_subscriber')

        self.bridge = CvBridge()
        self.pose_extractor = PoseExtractor()

        with open('src/cross_predictor/cross_predictor/config.yaml') as f:
            settings = yaml.load(f, Loader=yaml.SafeLoader)
        self.predictor_type = settings['PREDICTOR']

        image_sub = message_filters.Subscriber(self, Image, '/image_raw')
        detections_sub = message_filters.Subscriber(self, String, '/yolo/detections')
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, detections_sub], queue_size=10, slop=0.15, allow_headerless=True)
        self.sync.registerCallback(self.listener_callback)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST,
                         depth=10)
        self.publisher = self.create_publisher(Result, '/attention/resultv2', qos)
        self.get_logger().info('Subscribed to /yolo/image + /yolo/detections')

    def listener_callback(self, img_msg: Image, det_msg: String):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        data = json.loads(det_msg.data)
        attention_results = {}
        for det in data['detections']:
            xywh = np.array(det['xywh'], dtype=np.float32)
            track_id = det['track_id']

            orientation_ling, orientation_value, skeleton = self.pose_extractor.extract_pose(cv_image, xywh)
            att_ling, att_value = pedestrian_gaze(skeleton)
            if self.predictor_type != 'KG':
                att_ling = str(att_value)
                orientation_ling = str(orientation_value)
            attention_results[track_id] = [att_ling, orientation_ling]

        result = Result()
        result.header = img_msg.header
        result.header.stamp = img_msg.header.stamp
        result.result = attention_results.__str__()
        #self.get_logger().info(f"Attention results: {result.result}")
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
