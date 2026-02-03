import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import String
from my_msgs.msg import Result
import time
from collections import defaultdict
import threading # Added for safety

class ManualAggregator(Node):
    def __init__(self):
        super().__init__('manual_aggregator')
        
        self.storage = {}
        self.required_keys = {'action', 'attention', 'orientation'}
        self.max_age_seconds = 5.0  # Increased to be safe
        self.lock = threading.Lock() # Prevent race conditions

        
        
        self.pub = self.create_publisher(String, '/cross_predictor/final', 10)
        qos = QoSProfile(depth=10)
        # Explicitly named subscribers
        self.create_subscription(Result, '/action/resultv2', lambda msg: self._callback('action', msg), qos_profile=qos)
        self.create_subscription(Result, '/attention/resultv2', lambda msg: self._callback('attention', msg), qos_profile=qos)
        self.create_subscription(Result, '/orientation/resultv2', lambda msg: self._callback('orientation', msg), qos_profile=qos)

        self.buffer = defaultdict(dict)
        self.timeout_sec = 2.0  # drop incomplete sets after 1s

        # Optional: periodic cleanup timer to remove old frames
        self.create_timer(0.5, self._cleanup_buffer)
        self.get_logger().info("Fixed Manual Map Aggregator Online")

    def _callback(self, topic_name, msg):
        seq = msg.header.frame_id  # Use frame_id as sequence identifier
        self.buffer[seq][topic_name] = msg

        # Check if we have all 3 results
        if all(k in self.buffer[seq] for k in ['action', 'attention', 'orientation']):
            final_result_str = f"{self.buffer[seq]['action'].result}|{self.buffer[seq]['attention'].result}|{self.buffer[seq]['orientation'].result}"
            final_msg = String()
            final_msg.data = final_result_str
            self.pub.publish(final_msg)
            self.get_logger().info(f"Published final result for seq {seq}: {final_result_str}")

            # Remove the synced frame to avoid memory growth
            del self.buffer[seq]

    def _cleanup_buffer(self):
        # Drop any frames older than timeout_sec
        now = time.time()
        to_delete = []
        for seq, data in self.buffer.items():
            # Compare ROS time converted to seconds
            any_header = next(iter(data.values()))
            frame_time = any_header.header.stamp.sec + any_header.header.stamp.nanosec * 1e-9
            if now - frame_time > self.timeout_sec:
                to_delete.append(seq)
        for seq in to_delete:
            self.get_logger().warning(f"Dropping incomplete frame seq {seq} due to timeout")
            del self.buffer[seq]

def main():
    rclpy.init()
    node = ManualAggregator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()