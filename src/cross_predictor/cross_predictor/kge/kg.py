import numpy as np
import pandas as pd
import json
from ampligraph.evaluation import train_test_split_no_unseen

class PedestrianCrossKG():

    def __init__(self, settings):
        self.settings = settings
        self.included_features = settings['KG']['included_features']
        self.load_kg()
        self.create_triples()
        self.split_data()

    def load_kg(self):
        self.kg = pd.read_csv(self.settings['KG']['kg_file'])
        self.kg['train'] = self.kg.video is not None
        self.kg['ped_id'] =  self.kg.person
        self.kg['ped_id_frame'] =  self.kg.person+"_"+self.kg.frame.astype(str)
        self.kg['action'] = self.get_action(self.kg.action)
        self.kg['proximity'] = self.get_proximity(self.kg.proximity)
        self.kg['distance'] = self.get_distance(self.kg.distance)
        self.kg['orientation'] = self.get_orientation(self.kg.orientation)
        self.kg['attention'] = self.get_attention(self.kg.attention)
        self.kg['cross'] = self.get_cross(self.kg.cross)
        self.kg['fr'] = self.kg.frame
        self.kg['frame'] = self.kg.video + "_f" + self.kg.frame.astype(str)

        if self.settings['KG']['rules'] != 'None':
            f = open('kge/rules.json')
            data = json.load(f)
            rr = data[self.settings['KG']['rules']]['rules_explanation']
            self.rules = data[self.settings['KG']['rules']]['all_rules']
            self.kg['rules'] = self.get_rules(self.kg.proximity, self.kg.action, self.kg.distance, self.kg.orientation, self.kg.attention, rr)
        else:
            self.rules = []

        self.pedestrians = {}
        for index, row in self.kg.iterrows():
            ped = row['ped_id'] 
            if ped in self.pedestrians:
                self.pedestrians[ped].append(row['fr'])
            else:
                self.pedestrians[ped]=[row['fr']]

    def create_triples(self):
        self.triples = []
        self.triples.extend(self.rules)
        self.create_triples_from_dataset()
        self.triples_df = pd.DataFrame(self.triples, columns=["subject", "predicate", "object"])
        print(self.triples_df)

    def split_data(self):
        self.X_train, self.X_valid = train_test_split_no_unseen(np.array(self.triples), test_size=1000)
        print('Train set size: ', self.X_train.shape)
        print('Test set size: ', self.X_valid.shape)


    def create_triples_from_dataset(self):

        for _, row in self.kg[self.kg['train']].iterrows():

            if self.settings['KG']['rules'] != 'None':
                if self.settings['KG']['rules'] == 'jaad':
                    rule_features = self.get_rule_features_jaad ([row['proximity'], row['action'], row['distance'], row['orientation'], row['attention']])
            else:
                rule_features = []

            triples = []
            #Generalization
            triples.append(("Pedestrian", "hasChild", row['ped_id']))
            triples.append((row['ped_id_frame'], "instanceOf", row['ped_id']))
            prev_frame, pos_frame = self.get_prev_pos_relation(row['fr'], row['ped_id'])
            if prev_frame != 'NA':
                triples.append((prev_frame, "next", row['ped_id_frame']))

            if pos_frame != 'NA':
                triples.append((row['ped_id_frame'], "previous", pos_frame))

            if self.settings['KG']['rules'] != 'None':
                for r_feat in rule_features:
                    triples.append((row['ped_id_frame'], "Rule", r_feat))

            #Proximity information
            if 'Proximity' in self.included_features:
                triples.append((row['ped_id_frame'], "Location", row['proximity']))   
            #Action information
            if 'Action' in self.included_features:
                triples.append((row['ped_id_frame'], "Motion", row['action']))
            #Distance information
            if 'Distance' in self.included_features:
                triples.append((row['ped_id_frame'], "EgoDistance", row['distance']))
            #Orientation information
            if 'Orientation' in self.included_features:
                triples.append((row['ped_id_frame'], "Orientation", row['orientation']))
            #Attention information
            if 'Attention' in self.included_features:
                triples.append((row['ped_id_frame'], "Attention", row['attention']))
            #Cross information
            triples.append((row['ped_id_frame'], "Action", row['cross']))
            
            self.triples.extend(triples)



    
    def get_prev_pos_relation(self, frame, ped_id):
        actual = self.pedestrians[ped_id].index(frame)
        if actual != 0:
            prev_frame = ped_id+"_"+str(self.pedestrians[ped_id][actual-1])
        else:
            prev_frame = 'NA'

        if actual < len(self.pedestrians[ped_id])-1:
            pos_frame = ped_id+"_"+str(self.pedestrians[ped_id][actual+1])
        else:
            pos_frame = 'NA'
        
        return prev_frame, pos_frame
        

    def get_rules(self, l_proximity, l_action, l_distance, l_orientation, l_attention, base_probs):
        rules = []
        for index, value in enumerate(l_proximity):
            rule_features = self.get_rule_features_jaad ([l_proximity[index], l_action[index], l_distance[index], l_orientation[index], l_attention[index]])

            rules_str = "A"
            for feat in rule_features:
                if feat in base_probs:
                    rules_str = rules_str + ", " +feat
            
            rules.append(rules_str)
        
        return rules

    def get_rule_features_jaad (self, array_features):

        proximity = array_features[0]
        action = array_features[1]
        distance = array_features[2]
        orientation = array_features[3]
        attention = array_features[4]

        rule_features = []

        if action == 'Wave':
            rule_features.append('ped_wave')
        elif action == 'Stand':
            rule_features.append('ped_stand')
        elif action == "Run":
            if distance  == "TooNearToEgoVeh":
                rule_features.append('ped_tooNear_run')
        elif action == "Walk":
            if distance ==  "NearToEgoVeh":
                rule_features.append('ped_near_walk')


        if proximity == "MiddleDisFromCurb":
            rule_features.append("ped_pmoderate")
            if distance == "NearToEgoVeh":
                rule_features.append("ped_pmoderate_near")
            elif distance == "TooNearToEgoVeh":
                if attention == "NotLooking":
                    rule_features.append('ped_noLook_pmoderate_tooNear')
        elif proximity == "FarFromCurb":
            if attention == "NotLooking":
                rule_features.append('ped_noLook_pfar')

            if distance ==" NearToEgoVeh":
                rule_features.append('ped_pfar_near')

        elif proximity == "NearFromCurb":
            rule_features.append('ped_pnear')
            if action == "Stand":
                rule_features.append('ped_pnear_stand')
            elif action == "Wave":
                rule_features.append('ped_pnear_wave')
            elif action == "Walk":
                rule_features.append('ped_pnear_walk')

        if orientation == 'VehDirection':
            rule_features.append("ped_vehDir")

            if distance == "TooNearToEgoVeh":
                rule_features.append('ped_vehDir_tooNear')


        if attention == "NotLooking":
            if orientation == "RigthDirection":
                rule_features.append("ped_noLook_rigth")

                if action == "Walk":
                    rule_features.append('ped_noLook_rigth_walk')
                
                if distance == "NearToEgoVeh":
                    rule_features.append('ped_noLook_rigth_near')
                elif distance ==  "TooNearToEgoVeh":
                    rule_features.append('ped_noLook_rigth_tooNear')

            elif orientation == "VehDirection":
                if action == "Walk":
                    rule_features.append('ped_noLook_vehDir_walk')
                
                if distance == "TooNearToEgoVeh":
                    rule_features.append('ped_noLook_vehDir_tooNear')

            elif orientation == "LeftDirection":
                if action == "Run":
                    rule_features.append('ped_noLook_left_run')

                if distance == "NearToEgoVeh":
                    rule_features.append('ped_noLook_left_near')

            elif orientation == "OppositeVehDirection":
                if action == "Walk":
                    rule_features.append('ped_noLook_opposite_walk')

                if distance == "TooNearToEgoVeh":
                    rule_features.append("ped_noLook_opposite_tooNear")

            if distance == "NearToEgoVeh":
                if action == "Run":
                    rule_features.append('ped_noLook_near_run')
        
        elif attention == "Looking":
            if orientation == "OppositeVehDirection":
                rule_features.append('ped_look_opposite')

                if distance == "NearToEgoVeh":
                    rule_features.append('ped_look_opposite_near')

                    if action == "Walk":
                        rule_features.append('ped_look_near_walk')

                elif distance == "TooNearToEgoVeh":
                    rule_features.append('ped_look_opposite_tooNear')
            
            if distance == "TooNearToEgoVeh":
                if action == "Walk":
                    rule_features.append('ped_look_tooNear_walk')

        
        if orientation == "LeftDirection":
            if action == "Run":
                rule_features.append('ped_left_run')
            elif action == "Walk":
                if distance == "NearToEgoVeh":
                    rule_features.append('ped_left_near_walk')
            
            if distance ==  "NearToEgoVeh":
                rule_features.append("ped_left_near")
                if proximity == " NearToEgoVeh":
                    rule_features.append('ped_left_pnear_near')

        elif orientation == "RigthDirection":
            if distance  == "NearToEgoVeh":
                rule_features.append('ped_rigth_near')
                if action == "Walk":
                    rule_features.append('ped_rigth_near_walk')
            elif distance == "TooNearToEgoVeh":
                rule_features.append('ped_rigth_tooNear')
                if action == "Walk":
                    rule_features.append('ped_rigth_tooNear_walk')

            if action == "Walk":
                rule_features.append('ped_rigth_walk')

        elif orientation == "OppositeVehDirection":
            if action == "Run":
                rule_features.append('ped_opposite_run')
        
        return rule_features
    def get_action(self, actions):
        acts = []
        for action in actions:
            if action == 0 or action == '0':    
                acts.append('Stand')
            elif action == 1 or action == '1':
                acts.append('Walk')
            elif action == 2 or action == '2':
                acts.append('Wave')
            elif action == 3 or action == '3':
                acts.append('Run')
            elif action == 4 or action == '4':
                acts.append('Na')
            elif action == 'walking':
                acts.append('Walk')
            else:
                acts.append('Stand')
        return acts
    
    def get_orientation(self, orientations):
        acts = []
        for val_orientation in orientations:
            if val_orientation >=0 and val_orientation < 90:
                acts.append('VehDirection')
            elif val_orientation >= 90 and val_orientation <= 180:
                acts.append( 'LeftDirection')
            elif val_orientation > 180 and val_orientation < 270:
                acts.append('OppositeVehDirection')
            else:
                acts.append('RigthDirection')
        return acts
    
    def get_attention(self, attention):
        acts = []
        for att in attention:
            if att == 0 or att== '0':    
                acts.append('Looking')
            elif att == 1 or att=='1':
                acts.append('NotLooking')
            elif att == 'looking':    
                acts.append('Looking')
            elif att == 'not-looking':
                acts.append('NotLooking')
            
        return acts

    def get_proximity(self, proximities):
        prox = []
        for proximity in proximities:
            if proximity == 0:
                prox.append('NearFromCurb')
            elif proximity == 1:
                prox.append('MiddleDisFromCurb')
            else:
                prox.append('FarFromCurb')
        return prox
    
    def get_distance (self, distances):
        dis = []
        '''for distance in distances:
            if distance >= 0 and distance <= 0.9:
                dis.append('TooNearToEgoVeh')
            elif distance > 0.9 and distance <= 3.10:
                dis.append('NearToEgoVeh')
            elif distance > 3.10 and distance <= 5.30:
                dis.append('MiddleDisToEgoVeh')
            elif distance > 5.30 and distance <= 7.50:
                dis.append('FarToEgoVeh')
            else:
                dis.append('TooFarToEgoVeh')  '''
        for distance in distances:
            if distance >= 0 and distance <= 1.5:
                dis.append('TooNearToEgoVeh')
            elif distance > 1.5 and distance <= 6:
                dis.append('NearToEgoVeh')
            elif distance > 6 and distance <= 15:
                dis.append('MiddleDisToEgoVeh')
            elif distance > 15 and distance <= 20:
                dis.append('FarToEgoVeh')
            else:
                dis.append('TooFarToEgoVeh')    
        return dis  
    
    def get_cross (self, cross):
        cros = []
        for cr in cross:
            if cr == 'crossing':
                cros.append('CrossRoad')
            elif cr == 'not-crossing':
                cros.append('noCrossRoad')
        return cros
    
    def get_cross_value(self, cross):
        if cross == 'crossing' or cross == '1':
            return 1
        else:
            return 0