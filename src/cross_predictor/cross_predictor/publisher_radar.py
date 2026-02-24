import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random
from rclpy.qos import QoSProfile

class RadarSimulator(Node):
    def __init__(self):
        super().__init__('radar_simulator')
        self.declare_parameter('topic_name', '/radar/raw_data')
        self.declare_parameter('publish_period', 0.1)
        self.topic_name = self.get_parameter('topic_name').value
        self.publish_period = self.get_parameter('publish_period').value
        qos = QoSProfile(depth=50)
        self.publisher_ = self.create_publisher(String, self.topic_name, qos)
        # Match your camera rate (approx 1.0s) to keep things in sync
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info(f"Radar Simulator started on topic {self.topic_name}")
        self.seq = 0

    def timer_callback(self):
        self.seq += 1
        msg = String()
        # Simulate a pedestrian moving between 2m and 20m
        distance = random.uniform(2.0, 20.0)
        msg.data = str(distance)
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published radar distance: {distance:.2f}m (seq: {self.seq})")
        


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = RadarSimulator()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()