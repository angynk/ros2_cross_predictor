import rclpy
from message_filters import ApproximateTimeSynchronizer
import message_filters
from rclpy.node import Node
from my_msgs.msg import Result
from std_msgs.msg import  String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from collections import defaultdict
from rclpy.executors import MultiThreadedExecutor
import yaml
import ast
from cross_predictor.kge.kg_predictor import KGPredictor

class CrossPredictorAggregator(Node):
    def __init__(self):
        super().__init__('cross_predictor_aggregator')
        self.received_counts = {"action": 0, "attention": 0, "orientation": 0, "synced": 0, "proximity": 0}
        with open('/home/angie-melo/Documents/PHD/ROS_CARLA/CARLA_PROJECT/ped_predictor_ws/build/cross_predictor/cross_predictor/config.yaml') as f:
            settings = yaml.load(f, Loader=yaml.SafeLoader)
        self.predictor_kg = KGPredictor(settings)

        self.buffer = defaultdict(dict)
        self.timeout_sec = 3.0  # drop incomplete sets

        self.pub = self.create_publisher(String, '/cross_predictor/final', 10)
        qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT, # Don't wait for retries
                history=HistoryPolicy.KEEP_LAST,          # Only keep the newest
                depth=10                                   # Small buffer for lower latency
            )
        self.sub_action =message_filters.Subscriber(self, Result, '/action/resultv2', qos_profile=qos)
        self.sub_attention =message_filters.Subscriber(self, Result, '/attention/resultv2', qos_profile=qos)
        self.sub_proximity =message_filters.Subscriber(self, Result, '/proximity/resultv2',qos_profile=qos)

        self.ts = ApproximateTimeSynchronizer([self.sub_action, self.sub_attention, self.sub_proximity], queue_size=100,slop=2.0)
        self.ts.registerCallback(self.synchronized_callback)
        
        self.get_logger().info("Aggregator started. Waiting for synchronized Image headers...")

   
    def _increment_count(self, key):
        self.received_counts[key] += 1
        # Log every 20 messages so we don't spam the terminal
        if self.received_counts[key] % 20 == 0:
            self.get_logger().info(f"DEBUG: Received {key} messages: {self.received_counts[key]}")

    def synchronized_callback(self, msg_action, msg_attention, msg_proximity):
        # Retrieve the strings we hid in the frame_id
        self.received_counts["synced"] += 1
        self.get_logger().info(f"📊 COUNTS: {self.received_counts}")
        self._increment_count("action")
        action = msg_action.result
        self._increment_count("attention")
        attention = msg_attention.result
        #proximity = msg_proximity.header.frame_id
        self._increment_count("proximity")
        proximity = msg_proximity.result
        frame_features = self.parse_data(action, proximity,attention)
        if frame_features is None:
            self.get_logger().error("Mismatched IDs in synchronized messages. Skipping this set.")
            return
        self.get_logger().info(f"Extracted Features: {frame_features}")
        prediction, prob_cross, prob_nocross = self.predictor_kg.bayesian_method(frame_features["1"])
        self.get_logger().error("--- FINAL MERGE ---")
        final = String()
        final.data = f"PREDICTION={prediction} | PROB_CROSS={prob_cross:.2f} | PROB_NOCROSS={prob_nocross:.2f}"
        self.get_logger().info(f"Final Result: {final.data}")
        self.get_logger().error("-------------------")
        self.pub.publish(final)

    
    def clean_and_split(self, raw_str):
        # Remove brackets and split by ", " (comma + space) to isolate items
        return raw_str.strip("[]").split(", ")


    def parse_data(self, act_list, prox_list, att_list):
        result = {}

        # Helper function to split ID from the rest of the string
        # We use .split('-', 1) to ensure we only split at the first dash
        for item in self.clean_and_split(act_list):
            idx, val = item.split('-', 1)
            idx = idx.strip("'") 
            val = val.strip("'")  
            result[idx] = {"action": val}

        for item in self.clean_and_split(prox_list):
            idx, val = item.split('-', 1)
            idx = idx.strip("'") 
            val = val.strip("'")  
            if idx in result:
                result[idx]["proximity"] = val

        for item in self.clean_and_split(att_list):
            idx, rest = item.split('-', 1)
            idx = idx.strip("'") 
            rest = rest.strip("'")  
            attention, orientation = rest.split(',')
            if idx in result:
                result[idx]["attention"] = attention
                result[idx]["orientation"] = orientation

        return result

def main(args=None):
    rclpy.init()
    node = CrossPredictorAggregator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == '__main__':
    main()