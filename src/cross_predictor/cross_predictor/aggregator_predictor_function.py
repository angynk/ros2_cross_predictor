import rclpy
from message_filters import TimeSynchronizer
import message_filters
from rclpy.node import Node
from my_msgs.msg import Result
from std_msgs.msg import  String
from rclpy.qos import QoSProfile, ReliabilityPolicy
from collections import defaultdict
from rclpy.executors import MultiThreadedExecutor
import time

class CrossPredictorAggregator(Node):
    def __init__(self):
        super().__init__('cross_predictor_aggregator')
        self.received_counts = {"action": 0, "attention": 0, "orientation": 0, "synced": 0}

        self.buffer = defaultdict(dict)
        self.timeout_sec = 3.0  # drop incomplete sets

        self.pub = self.create_publisher(String, '/cross_predictor/final', 10)
        qos = QoSProfile(depth=50)
        self.sub_action =message_filters.Subscriber(self, Result, '/action/resultv2', qos_profile=qos)
        self.sub_attention =message_filters.Subscriber(self, Result, '/attention/resultv2', qos_profile=qos)
        #self.sub_proximity =message_filters.Subscriber(self, Image, '/proximity/result',qos_profile=10)
        self.sub_orientation =message_filters.Subscriber(self, Result, '/orientation/resultv2', qos_profile=qos)
        #self.sub_action.registerCallback(lambda msg: self._increment_count("action"))
        #self.sub_attention.registerCallback(lambda msg: self._increment_count("attention"))
        #self.sub_orientation.registerCallback(lambda msg: self._increment_count("orientation"))

        self.ts = TimeSynchronizer([self.sub_action, self.sub_attention, self.sub_orientation], queue_size=50)
        self.ts.registerCallback(self.synchronized_callback)
        
        self.get_logger().info("Aggregator started. Waiting for synchronized Image headers...")

   
    def _increment_count(self, key):
        self.received_counts[key] += 1
        # Log every 20 messages so we don't spam the terminal
        if self.received_counts[key] % 20 == 0:
            self.get_logger().info(f"DEBUG: Received {key} messages: {self.received_counts[key]}")

    def synchronized_callback(self, msg_action, msg_attention, msg_orientation):
        # Retrieve the strings we hid in the frame_id
        self.received_counts["synced"] += 1
        self.get_logger().info(f"📊 COUNTS: {self.received_counts}")
        self._increment_count("action")
        action = msg_action.result
        self._increment_count("attention")
        attention = msg_attention.result
        #proximity = msg_proximity.header.frame_id
        self._increment_count("orientation")
        orientation = msg_orientation.result
        
        self.get_logger().error("--- FINAL MERGE ---")
        self.get_logger().info(f"Action Result: {action}")
        self.get_logger().info(f"Attention Result:  {attention}")
        #self.get_logger().info(f"Proximity Result:  {proximity}")
        self.get_logger().info(f"Orientation Result:  {orientation}")
        self.get_logger().error("-------------------")
        final = String()
        final.data = f"ACTION={action} | ATTENTION={attention} | ORIENTATION={orientation}"
        self.pub.publish(final)

def main(args=None):
    rclpy.init()
    node = CrossPredictorAggregator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == '__main__':
    main()