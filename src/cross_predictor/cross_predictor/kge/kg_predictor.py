
import pandas as pd
import numpy as np
'''import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf'''
from ampligraph.utils import restore_model
from .kg import PedestrianCrossKG
from .bayes_probability import load_base_probabilities_jaad, load_base_probabilities_jaad_rules



class KGPredictor():

    def __init__(self, settings) :
        self.settings = settings
        self.included_features = self.settings['KG']['included_features']
        self.load_model()
        self.load_base_predictions()
        self.dt = {}

    # Load KGE Model trainned
    def load_model(self):
        self.model = restore_model(model_name_path= self.settings['KG']['model_path'])
        self.kg = PedestrianCrossKG(self.settings)
        self.model.calibrate(self.kg.X_train, 
                X_neg=None, 
                positive_base_rate=self.settings['KG']['positive_base_rate'], 
                batch_size=self.settings['KG']['batch'], 
                epochs=self.settings['KG']['epochs_calibration'], 
                verbose=0)
    
    # For execution time saving, load previosly all posibles triples evaluation
    def load_base_predictions(self):
        if self.settings['KG']['rules'] == 'None' and self.settings['KG']['dataset'] == 'jaad':
            self.base_probs = load_base_probabilities_jaad(self, self.settings['KG']['included_features'])
        if self.settings['KG']['rules'] == 'None' and self.settings['KG']['dataset'] == 'psi':
            self.base_probs = load_base_probabilities_jaad(self, self.settings['KG']['included_features'])
        elif self.settings['KG']['rules'] == 'jaad' and self.settings['KG']['dataset'] == 'jaad':
            self.base_probs = load_base_probabilities_jaad_rules(self)

    # Evaluate a triple using ampligraph
    def evaluate_triple(self, triple):
        triple_score = self.model.predict_proba(triple) 
        return triple_score

    # Predict an experiment 
    def predict_experiment(self, experiment_name):

        results = {}
        linguistic = {}
        labels = {'15f': [], '30f': [],  '60f': []}
        predictions = {'15f': [], '30f': [],  '60f': []}
        # OPEN THE CSV
        data = pd.read_csv(self.settings['DATA']['path']+experiment_name+'.csv', sep=';')
        gt_cross = 0
        for index, row in data.iterrows():
            frame_features = {'proximity': row.proximity, 'action': row.action, 'distance': row.distance,
                              'attention': row.attention, 'orientation': row.orientation}
            if gt_cross <= 90:
                prediction, pred_value, prob_cross, prob_nocross, linguistic_values = self.bayesian_method(frame_features)
                labels['15f'].append(self.kg.get_cross_value(row.prediction_15f))
                labels['30f'].append(self.kg.get_cross_value(row.prediction_30f))
                labels['60f'].append(self.kg.get_cross_value(row.prediction_60f))
                predictions['15f'].append(self.kg.get_cross_value(prediction))
                predictions['30f'].append(self.kg.get_cross_value(prediction))
                predictions['60f'].append(self.kg.get_cross_value(prediction))
                results[row.frame] = [row.experiment, row.id, row.frame, row.tstamp, row.attention, row.orientation, row.proximity,
                                    row.distance, row.action, row.cross,self.kg.get_cross_value(row.cross), row.prediction_15f, row.prediction_30f, row.prediction_60f,
                                    prediction, prob_cross]
                linguistic[row.frame] = [row.experiment, row.id, row.frame, row.tstamp, linguistic_values['attention'], linguistic_values['orientation'], linguistic_values['proximity'],
                                    linguistic_values['distance'], linguistic_values['action'], row.cross, row.prediction_15f, row.prediction_30f, row.prediction_60f,
                                    prediction, prob_cross]
                if row.cross == 'crossing':
                    gt_cross = gt_cross + 1
                
    
        return results, labels, predictions, linguistic

    # Compute softmax values for each sets of scores in x.
    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference
         
    # Calculate the crossing behavior from bayesian method using the KGE
    def bayesian_method(self, features):
        '''linguistic_values = {'proximity': self.kg.get_proximity([features['proximity']])[0],
                              'action': self.kg.get_action([features['action']])[0],
                              'distance': self.kg.get_distance([features['distance']])[0],
                              'orientation': self.kg.get_orientation([features['orientation']])[0],
                              'attention': self.kg.get_attention([features['attention']])[0],
                              'rules':''}'''
        linguistic_values = features
        #HYPHOTESIS IF NOT ACTION DON'T USE FOR PREDICTION

        hyphotesis_cross_score = self.base_probs['hyphotesis_cross']
        hyphotesis_ncross_score = self.base_probs['hyphotesis_ncross']
        evidence_triple_cross = []
        evidence_triple_ncross = []
        eviHyp_triple_cross = []
        eviHyp_triple_ncross = []

        if 'Proximity' in self.included_features:
            evidence_triple_cross.append(self.base_probs['evidence'][linguistic_values['proximity']])
            evidence_triple_ncross.append(self.base_probs['evidence'][linguistic_values['proximity']])
            eviHyp_triple_cross.append(self.base_probs['evidences_hypothesis_cross'][linguistic_values['proximity']])
            eviHyp_triple_ncross.append(self.base_probs['evidences_hypothesis_ncross'][linguistic_values['proximity']])
        if 'Action' in self.included_features:
            if features['action']== 'Stand' or features['action']== 'Walk':
                evidence_triple_cross.append(self.base_probs['evidence'][linguistic_values['action']])
                evidence_triple_ncross.append(self.base_probs['evidence'][linguistic_values['action']])
                eviHyp_triple_cross.append(self.base_probs['evidences_hypothesis_cross'][linguistic_values['action']])
                eviHyp_triple_ncross.append(self.base_probs['evidences_hypothesis_ncross'][linguistic_values['action']])
        if 'Distance' in self.included_features:
            evidence_triple_cross.append(self.base_probs['evidence'][linguistic_values['distance']])
            evidence_triple_ncross.append(self.base_probs['evidence'][linguistic_values['distance']])
            eviHyp_triple_cross.append(self.base_probs['evidences_hypothesis_cross'][linguistic_values['distance']])
            eviHyp_triple_ncross.append(self.base_probs['evidences_hypothesis_ncross'][linguistic_values['distance']])
        if 'Orientation' in self.included_features:
            evidence_triple_cross.append(self.base_probs['evidence'][linguistic_values['orientation']])
            evidence_triple_ncross.append(self.base_probs['evidence'][linguistic_values['orientation']])
            eviHyp_triple_cross.append(self.base_probs['evidences_hypothesis_cross'][linguistic_values['orientation']])
            eviHyp_triple_ncross.append(self.base_probs['evidences_hypothesis_ncross'][linguistic_values['orientation']])
        if 'Attention' in self.included_features:
            evidence_triple_cross.append(self.base_probs['evidence'][linguistic_values['attention']])
            evidence_triple_ncross.append(self.base_probs['evidence'][linguistic_values['attention']])
            eviHyp_triple_cross.append(self.base_probs['evidences_hypothesis_cross'][linguistic_values['attention']])
            eviHyp_triple_ncross.append(self.base_probs['evidences_hypothesis_ncross'][linguistic_values['attention']])
         
        if self.settings['KG']['rules'] != 'None' and self.settings['KG']['rules'] == 'jaad':
            bigger_score_rule = 0
            rules = self.kg.get_rule_features_jaad([linguistic_values['proximity'], linguistic_values['action'], linguistic_values['distance'], linguistic_values['orientation'], linguistic_values['attention']])

            for feat in rules:
                if feat in self.base_probs['evidence']:
                    
                    evidence_triple_ncross.append(self.base_probs['evidence'][feat])
                    eviHyp_triple_ncross.append(self.base_probs['evidences_hypothesis_ncross'][feat])
                    evidence_triple_cross.append(self.base_probs['evidence'][feat])
                    eviHyp_triple_cross.append(self.base_probs['evidences_hypothesis_cross'][feat])

                    if self.base_probs['evidences_hypothesis_ncross'][feat] >self.base_probs['evidences_hypothesis_cross'][feat]:
                        if self.base_probs['evidences_hypothesis_ncross'][feat] > bigger_score_rule:
                            bigger_score_rule = self.base_probs['evidences_hypothesis_ncross'][feat]
                            linguistic_values['rules'] = feat

                    else:
                        if self.base_probs['evidences_hypothesis_cross'][feat] > bigger_score_rule:
                            bigger_score_rule = self.base_probs['evidences_hypothesis_ncross'][feat]
                            linguistic_values['rules'] = feat

        #EVIDENCE 
        evidence_cross_score = np.prod(evidence_triple_cross)
        evidence_ncross_score = np.prod(evidence_triple_ncross)

        #EVIDENCE|HYPHOTESIS
        eviHyp_cross_score = np.prod(eviHyp_triple_cross)
        eviHyp_ncross_score = np.prod(eviHyp_triple_ncross)

        probability_cross = (hyphotesis_cross_score * eviHyp_cross_score) /evidence_cross_score
        probability_ncross = (hyphotesis_ncross_score * eviHyp_ncross_score) /evidence_ncross_score
        probabilities = [probability_cross[0], probability_ncross[0]]
        probabilities = self.softmax(probabilities)

        if probabilities[0] > probabilities[1]:
            return 'crossing',  probabilities[0], probabilities[1]
        else:
            return 'not-crossing' , probabilities[0], probabilities[1]
        
    