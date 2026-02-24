import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String, Float32
from my_msgs.msg import Result
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class DistanceCalculator(Node):
    def __init__(self):
        super().__init__('distance_calculator')
        self.declare_parameter('topic_name', '/radar/raw_data')
        self.topic_name = self.get_parameter('topic_name').value
        self.subscription = self.create_subscription(
            String, self.topic_name, self.calculate_callback, 10)
        
        with open('src/cross_predictor/cross_predictor/config.yaml') as f:
            settings = yaml.load(f, Loader=yaml.SafeLoader)
        self.predictor_type = settings['PREDICTOR']
        
        # This publishes to the topic your Aggregator is already listening to
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, # Don't wait for retries
                        history=HistoryPolicy.KEEP_LAST,          # Only keep the newest
                        depth=10)
        self.publisher = self.create_publisher(Result, '/distance/resultv2', qos)
        self.get_logger().info(f'Subscribed to {self.topic_name}')

    def calculate_callback(self, msg):
        raw_dist = float(msg.data)
        label = str(raw_dist)
        if self.predictor_type=='KG':
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
        #result.header = msg.header
        #result.header.stamp = msg.header.stamp
        result.result = f"['1-{label}']"
        self.publisher.publish(result)
        #self.get_logger().info(f"Calculated Distance: {raw_dist:.2f}m -> {label}")


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = DistanceCalculator()

    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()