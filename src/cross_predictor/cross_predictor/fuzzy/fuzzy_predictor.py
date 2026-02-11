import simpful  as sf
import pandas as pd
import json
import numpy as np

class FuzzyPredictor():
    def __init__(self, settings):
        self.settings = settings
        self.included_features = settings['FUZZY']['included_features']
        self.FS = sf.FuzzySystem()
        self.set_inputs_system(self.settings['FUZZY'])
        self.firing_strengths =  {}
        '''f = open(settings['FUZZY']['rules_detail_path'])
        data = json.load(f)
        self.rules = data['rules_features']'''
    

    def attention_input (self):
        #Attention Level Variable
        attention_l1 = sf.TriangleFuzzySet(-1.0,0.0,1.0,'not_looking')
        attention_l2 = sf.TriangleFuzzySet(0.0,1.0,2.0,'looking')
        self.FS.add_linguistic_variable("Attention", sf.LinguisticVariable([attention_l1, attention_l2], 
                                                                          universe_of_discourse=[-1, 2],concept="Attention Level"))
    def action_input (self):
        #Pedestrian Action Variable
        action_l1 = sf.TriangleFuzzySet(-1.0,0.0,1.0,'stand')
        action_l2 = sf.TriangleFuzzySet(0.0,1.0,2.0,'walk')
        action_l3 = sf.TriangleFuzzySet(1.0,2.0,3.0,'wave')
        action_l4 = sf.TriangleFuzzySet(2.0,3.0,4.0,'run')
        action_l5 = sf.TriangleFuzzySet(3.0,4.0,5.0,'na')
        self.FS. add_linguistic_variable("Action", sf.LinguisticVariable([action_l1, action_l2, action_l3, action_l4, action_l5], 
                                                                         universe_of_discourse=[-1, 5],  concept="Pedestrian Action"))

    def distance_car_input (self):
        #Distance to car Variable
        '''distance_l1 = sf.TriangleFuzzySet(-1.30,0.9,3.1025,'too_near')
        distance_l2 = sf.TriangleFuzzySet(0.9,3.10,5.30,'near')
        distance_l3 = sf.TriangleFuzzySet(3.10,5.30,7.50,'moderate')
        distance_l4 = sf.TriangleFuzzySet(5.30,7.50,9.71,'far')
        distance_l5 = sf.TriangleFuzzySet(7.50,9.71,11.91,'too_far')'''
        distance_l1 = sf.TriangleFuzzySet(-1,1.5,3.0,'too_near')
        distance_l2 = sf.TriangleFuzzySet(1.5,3.0,6.0,'near')
        distance_l3 = sf.TriangleFuzzySet(3.0,6.0,15.0,'moderate')
        distance_l4 = sf.TriangleFuzzySet(6.0,15.0,20.0,'far')
        distance_l5 = sf.TriangleFuzzySet(15.0,20.0,54.0,'too_far')
        self.FS.add_linguistic_variable("CDistance", sf.LinguisticVariable([distance_l1, distance_l2, distance_l3,distance_l4,distance_l5], 
                                                                           universe_of_discourse=[-2, 12],  concept="Pedestrian Distance to Car"))

    def orientation_input (self):
        orientation_l1 = sf.TriangleFuzzySet(-86.13,3.02,92.17,'car_direction')
        orientation_l2 = sf.TriangleFuzzySet(3.02,92.17,181.32,'left')
        orientation_l3 = sf.TriangleFuzzySet(92.17,181.32,270.47,'opposite_car_direction')
        orientation_l4 = sf.TriangleFuzzySet(181.32,270.47,359.62,'right')
        orientation_l5 = sf.TriangleFuzzySet(270.47,359.62,448.77,'right')
        self.FS.add_linguistic_variable("Orientation", sf.LinguisticVariable([orientation_l1,orientation_l2, orientation_l3, orientation_l4,orientation_l5 ],
                                                                             universe_of_discourse=[0, 360],  concept="Pedestrian Orientation"))

    def proximity_road_input (self):
        proximity_l1 = sf.TriangleFuzzySet(-1.0,0.0,1.0,'near')
        proximity_l2 = sf.TriangleFuzzySet(0.0,1.0,2.0,'moderate')
        proximity_l3 = sf.TriangleFuzzySet(1.0,2.0,3.0,'far')
        self.FS.add_linguistic_variable("Proximity", sf.LinguisticVariable([proximity_l1,proximity_l2, proximity_l3],
                                                                            universe_of_discourse=[-1, 3],  concept="Pedestrian Proximity to road"))
       

    def set_outuput_zero_sugeno (self):
        #Crossing crisp values
        self.FS.set_crisp_output_value("crossing", 1)
        self.FS.set_crisp_output_value("not_crossing", 0)
    
    
    def set_inputs_system (self,rules):
        if 'Attention' in self.included_features:
            self.attention_input()
        if 'Orientation' in self.included_features:
            self.orientation_input()
        if 'Proximity' in self.included_features:
            self.proximity_road_input()
        if 'Distance' in self.included_features:
            self.distance_car_input()
        if 'Action' in self.included_features:
            self.action_input()
        self.set_outuput_zero_sugeno()
        self.FS.add_rules_from_file(self.settings['FUZZY']['rules_path'])


    def predict_action(self, features):

        self.features_relevance= {"Attention": 0, "Orientation":0, "Proximity": 0, "CDistance":0, "Action":0}

        if 'Attention' in self.included_features:
            self.FS.set_variable("Attention", float(features["attention"]))
        if 'Orientation' in self.included_features:
            self.FS.set_variable("Orientation", float(features["orientation"]))
        if 'Proximity' in self.included_features:
            self.FS.set_variable("Proximity", int(features["proximity"]))
        if 'Distance' in self.included_features:
            self.FS.set_variable("CDistance", float(features["distance"]))
        if 'Action' in self.included_features:
            self.FS.set_variable("Action", int(features['action']))
        pred = round(self.FS.Sugeno_inference(["Cross"])['Cross'],2)
        
        # TO GET ACTIVATED RULES

        '''activated_rules = self.FS.get_firing_strengths()
        for x,strength in enumerate(activated_rules):
            if strength > 0 :
                counter = 1
            else:
                counter = 0
            if self.firing_strengths.get(str(x)) is None:
                self.firing_strengths[str(x)] = {'strength' : strength, 'counter': counter}
            else:
                if strength > self.firing_strengths[str(x)]['strength'] :
                    self.firing_strengths[str(x)]['strength'] = strength
                self.firing_strengths[str(x)]['counter'] = self.firing_strengths[str(x)]['counter'] + counter
            #self.get_features_relevance(str(x), strength)'''
        
        #f_relevance = self.softmax([self.features_relevance["Attention"],self.features_relevance["Orientation"],self.features_relevance["Proximity"],self.features_relevance["CDistance"],self.features_relevance["Action"]] )
        
        return pred,  self.transform_cross_prob(pred)
    
    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference
    
    def get_features_relevance(self, rule, strength):
        rule = int(rule) + 1
        rules_features = self.rules[str(rule)]["features"]
        for r_f in rules_features:
            self.features_relevance[r_f] = self.features_relevance[r_f] + strength

        
    def activation_rules(self):
        for rule_name, firing_strength in self.firing_strengths.items():
            print('Rule No.'+str(rule_name)+' Counter:'+str(firing_strength['counter'])+' - Strength:'+str(firing_strength['strength']))

    # Predict an experiment 
    def predict_experiment(self, experiment_name):

        results = {}
        labels = {'15f': [], '30f': [],  '60f': []}
        predictions = {'15f': [], '30f': [],  '60f': []}
        # OPEN THE CSV
        gt_cross = 0
        data = pd.read_csv(self.settings['DATA']['path']+experiment_name+'.csv', sep=';')
        for index, row in data.iterrows():
            frame_features = {'proximity': row.proximity, 'action': row.action, 'distance': row.distance,
                              'attention': row.attention, 'orientation': row.orientation}
            if gt_cross <= 90:
                prediction_prob, features_relevance = self.predict_action(frame_features)
                labels['15f'].append(self.get_cross_value(row.prediction_15f))
                labels['30f'].append(self.get_cross_value(row.prediction_30f))
                labels['60f'].append(self.get_cross_value(row.prediction_60f))
                prediction_label, prediction_val = self.transform_cross_prob(prediction_prob)
                predictions['15f'].append(prediction_val)
                predictions['30f'].append(prediction_val)
                predictions['60f'].append(prediction_val)
                results[row.frame] = [prediction_label, prediction_prob, features_relevance[0], features_relevance[1], features_relevance[2], features_relevance[3], features_relevance[4]]
            
            if row.cross == 'crossing':
                    gt_cross = gt_cross + 1
        
        return results, labels, predictions
    
    def transform_action(self, action):
        if action=='Walk':
            return 1 
        return 0
    
    def transform_attention(self, attention):
        if attention == 'Looking':
            return 1
        return 0
    
    def transform_proximity(self, proximity):
        if proximity == 'NearFromCurb':
            return 0
        elif proximity == 'MiddleFromCurb':
            return 1
        else:
            return 2
        
    def transform_cross_prob(self, cross):
        if cross >= 0.50:
            return 'crossing'
        else:
            return 'not-crossing'
    
    def get_cross_value(self, cross):
        if cross == 'crossing' or cross == '1':
            return 1
        else:
            return 0
    