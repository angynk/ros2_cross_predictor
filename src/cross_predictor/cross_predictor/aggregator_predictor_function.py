import rclpy
from message_filters import ApproximateTimeSynchronizer
import message_filters
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import  String
from collections import defaultdict
import time

class CrossPredictorAggregator(Node):
    def __init__(self):
        super().__init__('cross_predictor_aggregator')

        self.buffer = defaultdict(dict)
        self.timeout_sec = 1.0  # drop incomplete sets

        self.pub = self.create_publisher(String, '/cross_predictor/final', 10)

        self.sub_action =message_filters.Subscriber(self, Image, '/action/result', qos_profile=10)
        self.sub_attention =message_filters.Subscriber(self, Image, '/attention/result',qos_profile=10)

        self.ts = ApproximateTimeSynchronizer([self.sub_action, self.sub_attention], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synchronized_callback)
        
        self.get_logger().info("Aggregator started. Waiting for synchronized Image headers...")

    
    def synchronized_callback(self, msg_a, msg_b):
        # Retrieve the strings we hid in the frame_id
        res_a = msg_a.header.frame_id
        res_b = msg_b.header.frame_id
        
        self.get_logger().error("--- FINAL MERGE ---")
        self.get_logger().info(f"Object Result: {res_a}")
        self.get_logger().info(f"Color Result:  {res_b}")
        self.get_logger().error("-------------------")
        final = String()
        final.data = f"ACTION={res_a} | ATTENTION={res_b}"
        self.pub.publish(final)

def main(args=None):
    rclpy.init(args=args)
    aggregator = CrossPredictorAggregator()
    
    try:
        rclpy.spin(aggregator)
    except KeyboardInterrupt:
        pass
    finally:
        aggregator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()