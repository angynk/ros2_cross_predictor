import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cross_predictor.features_extractor.yolov_detector import YOLOVDetector


class DetectorNode(Node):
    """Single YOLO tracking node. Publishes detections so all feature
    subscribers share the same bounding boxes and track IDs per frame."""

    def __init__(self):
        super().__init__('yolo_detector')
        self.declare_parameter('topic_name', '/image_raw')
        self.topic_name = self.get_parameter('topic_name').value

        self.bridge = CvBridge()
        self.yolov_detector = YOLOVDetector()
        self.frame_number = 0

        qos = QoSProfile(depth=50)
        self.subscription = self.create_subscription(
            Image, self.topic_name, self.listener_callback, 10)
        # Re-publish the original image so subscribers can sync to it
        self.image_pub = self.create_publisher(Image, '/yolo/image', qos)
        self.detections_pub = self.create_publisher(String, '/yolo/detections', qos)
        self.get_logger().info(f'Detector subscribed to {self.topic_name}')

    def listener_callback(self, msg: Image):
        self.frame_number += 1
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.yolov_detector.track_pedestrians(cv_image)

        detections = []
        for result in results:
            if result.boxes.id is not None:
                ids = result.boxes.id.numpy()
                for i in range(len(ids)):
                    xywh = result.boxes.xywh[i].cpu().numpy().tolist()
                    xyxy = result.boxes.xyxy[i].cpu().numpy().tolist()
                    unique_id = self.yolov_detector.resolve_tracking_id(
                        int(ids[i]), self.frame_number)
                    detections.append({
                        'track_id': unique_id,
                        'xywh': xywh,
                        'xyxy': xyxy,
                    })

        det_msg = String()
        det_msg.data = json.dumps({
            'stamp_sec': msg.header.stamp.sec,
            'stamp_nanosec': msg.header.stamp.nanosec,
            'detections': detections,
        })
        self.image_pub.publish(msg)
        self.detections_pub.publish(det_msg)
        if len(detections) > 0:
            count = self.yolov_detector.pedestrian_count
            self.get_logger().info(f'Total pedestrians identified: {count}')


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
