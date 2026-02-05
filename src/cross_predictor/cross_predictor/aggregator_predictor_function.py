import rclpy
from message_filters import TimeSynchronizer
import message_filters
from rclpy.node import Node
from my_msgs.msg import Result
from std_msgs.msg import  String
from rclpy.qos import QoSProfile, ReliabilityPolicy
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
        qos = QoSProfile(depth=50)
        self.sub_action =message_filters.Subscriber(self, Result, '/action/resultv2', qos_profile=qos)
        self.sub_attention =message_filters.Subscriber(self, Result, '/attention/resultv2', qos_profile=qos)
        self.sub_proximity =message_filters.Subscriber(self, Result, '/proximity/resultv2',qos_profile=qos)
        #self.sub_orientation =message_filters.Subscriber(self, Result, '/orientation/resultv2', qos_profile=qos)
        #self.sub_action.registerCallback(lambda msg: self._increment_count("action"))
        #self.sub_attention.registerCallback(lambda msg: self._increment_count("attention"))
        #self.sub_orientation.registerCallback(lambda msg: self._increment_count("orientation"))

        self.ts = TimeSynchronizer([self.sub_action, self.sub_attention, self.sub_proximity], queue_size=50)
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
        frame_features = self.merge_results(action, attention, proximity)
        if frame_features is None:
            self.get_logger().error("Mismatched IDs in synchronized messages. Skipping this set.")
            return
        self.get_logger().info(f"Extracted Features: {frame_features}")
        prediction, prob_cross, prob_nocross = self.predictor_kg.bayesian_method(frame_features[1])
        self.get_logger().error("--- FINAL MERGE ---")
        final = String()
        final.data = f"ACTION={action} | ATTENTION={attention} | PROXIMITY={proximity} | PREDICTION={prediction} | PROB_CROSS={prob_cross:.2f} | PROB_NOCROSS={prob_nocross:.2f}"
        self.get_logger().info(f"Final Result: {final.data}")
        self.get_logger().error("-------------------")
        self.pub.publish(final)

    def extract_id_and_val(self,raw_str):
        """Turns \"['1-Na']\" into (1, 'Na')"""
        # 1. Convert string representation of list to actual list
        data_list = ast.literal_eval(raw_str)
        # 2. Get the first element '1-Na'
        inner_str = data_list[0]
        # 3. Split by the first hyphen only
        id_part, val_part = inner_str.split('-', 1)
        return int(id_part), val_part
    
    def merge_results(self,act, att, prox):
        # Extract data from all three
        id_a, val_a = self.extract_id_and_val(act)
        id_p, val_att = self.extract_id_and_val(att)
        id_d, val_prox = self.extract_id_and_val(prox)
        
        # Sanity check: ensure they are all for the same ID
        if id_a == id_p == id_d:
            return {
                id_a: {
                    'proximity': val_prox,
                    'action': val_a,
                    'attention': val_att,
                    'orientation': "LeftDirection",
                    'distance': "MiddleDisToEgoVeh"
                }
            }
        else:
            return None # Or handle the mismatch logic

def main(args=None):
    rclpy.init()
    node = CrossPredictorAggregator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == '__main__':
    main()