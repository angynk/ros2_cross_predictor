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
import json
import logging
from cross_predictor.kge.kg_predictor import KGPredictor
from cross_predictor.fuzzy.fuzzy_predictor import FuzzyPredictor

class CrossPredictorAggregator(Node):
    def __init__(self):
        super().__init__('cross_predictor_aggregator')
        self.received_counts = {"action": 0, "attention": 0, "orientation": 0, "synced": 0, "proximity": 0}
        with open('src/cross_predictor/cross_predictor/config.yaml') as f:
            settings = yaml.load(f, Loader=yaml.SafeLoader)
        self.predictor_type = settings['PREDICTOR']
        if self.predictor_type=='KG':
            self.predictor_kg = KGPredictor(settings)
        else:
            self.predictor_fuzzy = FuzzyPredictor(settings)

        self.latest_distance_label = "Unknown"
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

        '''self.sub_distance= self.create_subscription(
            Result, 
            '/distance/resultv2', 
            self.distance_callback, 
            qos_profile=qos)'''

        self.ts = ApproximateTimeSynchronizer([self.sub_action, self.sub_attention, self.sub_proximity], queue_size=100,slop=0.15)
        self.ts.registerCallback(self.synchronized_callback)
        self.logger = logging.getLogger('CrossPredictorAggregator')
        file_handler = logging.FileHandler('detailed_log.log')
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', 
                               datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.get_logger().info("Aggregator started. Waiting for synchronized Image headers...")

    def distance_callback(self, msg):
        # Update the global state whenever radar data arrives
        # Extract 'MiddleDisToEgoVeh' from "['1-MiddleDisToEgoVeh']"
        try: 
            self.latest_distance_label = ast.literal_eval(msg.result)[0].split('-')[1]
            if self.predictor_type!='KG':
                self.latest_distance_label = float(self.latest_distance_label)
        except:
            pass
   
    def _increment_count(self, key):
        self.received_counts[key] += 1

    def synchronized_callback(self, msg_action, msg_attention, msg_proximity):
        output = {"frame_id": msg_action.header.frame_id, "pedestrians": {} ,
                   "synced": self.received_counts["synced"], 
                  } 
        self.received_counts["synced"] += 1
        #self.logger.info(f"📊 COUNTS: {self.received_counts}")
        self._increment_count("action")
        action = msg_action.result
        self._increment_count("attention")
        attention = msg_attention.result
        self._increment_count("proximity")
        proximity = msg_proximity.result
        frame_features = self.parse_data(action, proximity,attention)
        if len(frame_features) == 0:
            #self.logger.info(f"Action {action}. Attention: {attention}. Proximity: {proximity}")
            self.logger.info(output)
            self.logger.info("-------------------")
            #self.logger.error("Pedestrian Not Found. Skipping this set.")
            return
        '''for key in frame_features:
            frame_features[key]["distance"] = self.latest_distance_label'''
        final = String()
        for key in frame_features:
            output["pedestrians"][key] = {"prediction": "", "crossing_probability": 0.0, "features": frame_features[key]}
            if self.predictor_type=='KG':
                prediction, prob_cross, prob_nocross = self.predictor_kg.bayesian_method(frame_features[key])
                final.data = f"PREDICTION={prediction} | PROB_CROSS={prob_cross:.2f} | PROB_NOCROSS={prob_nocross:.2f}"
            else:
                prob_cross, prediction = self.predictor_fuzzy.predict_action(frame_features[key])
                final.data = f"PREDICTION={prediction} | PROB_CROSS={prob_cross:.2f} "
            output["pedestrians"][key]["prediction"] = prediction
            output["pedestrians"][key]["crossing_probability"] = prob_cross
        
        final.data = str(output)
        self.logger.info(output)
        self.logger.info("-------------------")
        self.pub.publish(final)

    
    def clean_and_split(self, raw_str):
        # Remove brackets and split by ", " (comma + space) to isolate items
        return raw_str.strip("[]").split(", ")


    def parse_data(self, act_list, prox_list, att_list):
        result = {}
        actions = json.loads(act_list.replace("'", "\""))
        proximities = json.loads(prox_list.replace("'", "\""))
        attentions = json.loads(att_list.replace("'", "\""))
        for key in actions.keys():
            if key in proximities and key in attentions:
                result[key] = {"action": actions[key], "proximity": proximities[key], "attention": attentions[key][0],
                                "orientation": attentions[key][1]}
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