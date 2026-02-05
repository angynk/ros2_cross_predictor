import numpy as np

#################################################################    JAAD TRIPLES  ####################################################################################
def load_base_probabilities_jaad(bayes, included_features):
    base_probs = {'hyphotesis_cross': bayes.evaluate_triple(np.array([['Pedestrian', 'Action', 'CrossRoad']])), 
                    'hyphotesis_ncross': bayes.evaluate_triple(np.array([['Pedestrian', 'Action', 'noCrossRoad']])),
                    'evidence':{},
                    'evidences_hypothesis_cross': {},
                    'evidences_hypothesis_ncross': {}
                    }
    
    if 'Attention' in included_features:
        base_probs['evidence']['Looking'] = bayes.evaluate_triple(np.array([['Pedestrian', 'Attention', 'Looking']]))
        base_probs['evidences_hypothesis_cross']['Looking'] = bayes.evaluate_triple(np.array([['Looking', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['Looking'] = bayes.evaluate_triple(np.array([['Looking', 'Action', 'noCrossRoad']]))
        base_probs['evidence']['NotLooking'] = bayes.evaluate_triple(np.array([['Pedestrian', 'Attention', 'NotLooking']]))
        base_probs['evidences_hypothesis_cross']['NotLooking'] = bayes.evaluate_triple(np.array([['NotLooking', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['NotLooking'] = bayes.evaluate_triple(np.array([['NotLooking', 'Action', 'noCrossRoad']]))
        
    if 'Orientation' in included_features:
        base_probs['evidence']['VehDirection'] = bayes.evaluate_triple(np.array([['Pedestrian', 'Orientation', 'VehDirection']]))
        base_probs['evidences_hypothesis_cross']['VehDirection'] = bayes.evaluate_triple(np.array([['VehDirection', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['VehDirection'] = bayes.evaluate_triple(np.array([['VehDirection', 'Action', 'noCrossRoad']]))
        base_probs['evidence']['LeftDirection'] = bayes.evaluate_triple(np.array([['Pedestrian', 'Orientation', 'LeftDirection']]))
        base_probs['evidences_hypothesis_cross']['LeftDirection'] = bayes.evaluate_triple(np.array([['LeftDirection', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['LeftDirection'] = bayes.evaluate_triple(np.array([['LeftDirection', 'Action', 'noCrossRoad']]))
        base_probs['evidence']['OppositeVehDirection'] = bayes.evaluate_triple(np.array([['Pedestrian', 'Orientation', 'OppositeVehDirection']]))
        base_probs['evidences_hypothesis_cross']['OppositeVehDirection'] = bayes.evaluate_triple(np.array([['OppositeVehDirection', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['OppositeVehDirection'] = bayes.evaluate_triple(np.array([['OppositeVehDirection', 'Action', 'noCrossRoad']]))
        base_probs['evidence']['RigthDirection'] = bayes.evaluate_triple(np.array([['Pedestrian', 'Orientation', 'RigthDirection']]))
        base_probs['evidences_hypothesis_cross']['RigthDirection'] = bayes.evaluate_triple(np.array([['RigthDirection', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['RigthDirection'] = bayes.evaluate_triple(np.array([['RigthDirection', 'Action', 'noCrossRoad']]))

    if 'Proximity' in included_features:
        base_probs['evidence']['NearFromCurb'] = bayes.evaluate_triple(np.array([['Pedestrian', 'Location', 'NearFromCurb']]))
        base_probs['evidences_hypothesis_cross']['NearFromCurb'] = bayes.evaluate_triple(np.array([['NearFromCurb', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['NearFromCurb'] = bayes.evaluate_triple(np.array([['NearFromCurb', 'Action', 'noCrossRoad']]))
        base_probs['evidence']['MiddleDisFromCurb'] = bayes.evaluate_triple(np.array([['Pedestrian', 'Location', 'MiddleDisFromCurb']]))
        base_probs['evidences_hypothesis_cross']['MiddleDisFromCurb'] = bayes.evaluate_triple(np.array([['MiddleDisFromCurb', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['MiddleDisFromCurb'] = bayes.evaluate_triple(np.array([['MiddleDisFromCurb', 'Action', 'noCrossRoad']]))
        base_probs['evidence']['FarFromCurb'] = bayes.evaluate_triple(np.array([['Pedestrian', 'Location', 'FarFromCurb']]))
        base_probs['evidences_hypothesis_cross']['FarFromCurb'] = bayes.evaluate_triple(np.array([['FarFromCurb', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['FarFromCurb'] = bayes.evaluate_triple(np.array([['FarFromCurb', 'Action', 'noCrossRoad']]))

    if 'Distance' in included_features:
        base_probs['evidence']['TooNearToEgoVeh'] = bayes.evaluate_triple(np.array([['Pedestrian', 'EgoDistance', 'TooNearToEgoVeh']]))
        base_probs['evidences_hypothesis_cross']['TooNearToEgoVeh'] = bayes.evaluate_triple(np.array([['TooNearToEgoVeh', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['TooNearToEgoVeh'] = bayes.evaluate_triple(np.array([['TooNearToEgoVeh', 'Action', 'noCrossRoad']]))
        base_probs['evidence']['NearToEgoVeh'] = bayes.evaluate_triple(np.array([['Pedestrian', 'EgoDistance', 'NearToEgoVeh']]))
        base_probs['evidences_hypothesis_cross']['NearToEgoVeh'] = bayes.evaluate_triple(np.array([['NearToEgoVeh', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['NearToEgoVeh'] = bayes.evaluate_triple(np.array([['NearToEgoVeh', 'Action', 'noCrossRoad']]))
        base_probs['evidence']['MiddleDisToEgoVeh'] = bayes.evaluate_triple(np.array([['Pedestrian', 'EgoDistance', 'MiddleDisToEgoVeh']]))
        base_probs['evidences_hypothesis_cross']['MiddleDisToEgoVeh'] = bayes.evaluate_triple(np.array([['MiddleDisToEgoVeh', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['MiddleDisToEgoVeh'] = bayes.evaluate_triple(np.array([['MiddleDisToEgoVeh', 'Action', 'noCrossRoad']]))
        base_probs['evidence']['FarToEgoVeh'] = bayes.evaluate_triple(np.array([['Pedestrian', 'EgoDistance', 'FarToEgoVeh']]))
        base_probs['evidences_hypothesis_cross']['FarToEgoVeh'] = bayes.evaluate_triple(np.array([['FarToEgoVeh', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['FarToEgoVeh'] = bayes.evaluate_triple(np.array([['FarToEgoVeh', 'Action', 'noCrossRoad']]))
        base_probs['evidence']['TooFarToEgoVeh'] = bayes.evaluate_triple(np.array([['Pedestrian', 'EgoDistance', 'TooFarToEgoVeh']]))
        base_probs['evidences_hypothesis_cross']['TooFarToEgoVeh'] = bayes.evaluate_triple(np.array([['TooFarToEgoVeh', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['TooFarToEgoVeh'] = bayes.evaluate_triple(np.array([['TooFarToEgoVeh', 'Action', 'noCrossRoad']]))
    
    if 'Action' in included_features:
        base_probs['evidence']['Stand'] = bayes.evaluate_triple(np.array([['Pedestrian', 'Motion', 'Stand']]))
        base_probs['evidences_hypothesis_cross']['Stand'] = bayes.evaluate_triple(np.array([['Stand', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['Stand'] = bayes.evaluate_triple(np.array([['Stand', 'Action', 'noCrossRoad']]))
        base_probs['evidence']['Walk'] = bayes.evaluate_triple(np.array([['Pedestrian', 'Motion', 'Walk']]))
        base_probs['evidences_hypothesis_cross']['Walk'] = bayes.evaluate_triple(np.array([['Walk', 'Action', 'CrossRoad']]))
        base_probs['evidences_hypothesis_ncross']['Walk'] = bayes.evaluate_triple(np.array([['Walk', 'Action', 'noCrossRoad']]))
        
    return base_probs


# Triples with rules from JAAD
def load_base_probabilities_jaad_rules(bayes):
    base_probs = {
                           'hyphotesis_cross': bayes.evaluate_triple(np.array([['Pedestrian', 'Action', 'CrossRoad']])), 
                           'hyphotesis_ncross': bayes.evaluate_triple(np.array([['Pedestrian', 'Action', 'noCrossRoad']])), 
                        
                           'evidence':{
                               'NearFromCurb': bayes.evaluate_triple(np.array([['Pedestrian', 'Location', 'NearFromCurb']])),
                               'MiddleDisFromCurb': bayes.evaluate_triple(np.array([['Pedestrian', 'Location', 'MiddleDisFromCurb']])),
                               'FarFromCurb': bayes.evaluate_triple(np.array([['Pedestrian', 'Location', 'FarFromCurb']])),
                               'Stand': bayes.evaluate_triple(np.array([['Pedestrian', 'Motion', 'Stand']])),
                               'Walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Motion', 'Walk']])),
                               'VehDirection': bayes.evaluate_triple(np.array([['Pedestrian', 'Orientation', 'VehDirection']])),
                               'LeftDirection': bayes.evaluate_triple(np.array([['Pedestrian', 'Orientation', 'LeftDirection']])),
                               'OppositeVehDirection': bayes.evaluate_triple(np.array([['Pedestrian', 'Orientation', 'OppositeVehDirection']])),
                               'RigthDirection': bayes.evaluate_triple(np.array([['Pedestrian', 'Orientation', 'RigthDirection']])),
                               'Looking': bayes.evaluate_triple(np.array([['Pedestrian', 'Attention', 'Looking']])),
                               'NotLooking': bayes.evaluate_triple(np.array([['Pedestrian', 'Attention', 'NotLooking']])),
                               'TooNearToEgoVeh': bayes.evaluate_triple(np.array([['Pedestrian', 'EgoDistance', 'TooNearToEgoVeh']])),
                               'NearToEgoVeh': bayes.evaluate_triple(np.array([['Pedestrian', 'EgoDistance', 'NearToEgoVeh']])),
                               'MiddleDisToEgoVeh': bayes.evaluate_triple(np.array([['Pedestrian', 'EgoDistance', 'MiddleDisToEgoVeh']])),
                               'FarToEgoVeh': bayes.evaluate_triple(np.array([['Pedestrian', 'EgoDistance', 'FarToEgoVeh']])),
                               'TooFarToEgoVeh': bayes.evaluate_triple(np.array([['Pedestrian', 'EgoDistance', 'TooFarToEgoVeh']])),
                               'Ped_wave': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.62']])),
                               'Ped_stand': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.75']])),
                               'Ped_pmoderate': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.63']])),
                               'Ped_rigth': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.50']])),
                               'Ped_vehDir': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.70']])),
                               'Ped_noLook_pfar': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.69']])),
                               'Ped_noLook_rigth': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.48']])),
                               'Ped_noLook_vehDir_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.69']])),
                               'Ped_noLook_vehDir_tooNear': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.78']])),
                               'Ped_noLook_rigth_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.48']])),
                               'Ped_noLook_rigth_near': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.49']])),
                               'Ped_look_opposite': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.62']])),
                               'Ped_look_opposite_near': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.66']])),
                               'Ped_look_near_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.58']])),
                               'Ped_vehDir_tooNear': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.78']])),
                               'Ped_left_run': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.55']])),
                               'Ped_left_near': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.55']])),
                               'Ped_left_near_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.53']])),
                               'Ped_rigth_near_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.47']])),
                               'Ped_rigth_tooNear': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.55']])),
                               'Ped_pnear_stand': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.65']])),
                               'Ped_pmoderate_near': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.70']])),
                               'Ped_pfar_near': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'NoCross_0.80']])),
                               'Ped_pnear': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.65']])),
                               'Ped_noLook_left_run': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.56']])),
                               'Ped_noLook_left_near': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.47']])),
                               'Ped_noLook_opposite_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.68']])),
                               'Ped_noLook_opposite_tooNear': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.69']])),
                               'Ped_noLook_rigth_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.58']])),
                               'Ped_noLook_rigth_tooNear': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.58']])),
                               'Ped_noLook_rigth_near': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.52']])),
                               'Ped_noLook_pmoderate_tooNear': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.50']])),
                               'Ped_noLook_near_run': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.54']])),
                               'Ped_look_opposite_tooNear': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.49']])),
                               'Ped_look_tooNear_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.50']])),
                               'Ped_left_near': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.48']])),
                               'Ped_left_pnear_near': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.63']])),
                               'Ped_left_near_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.48']])),
                               'Ped_opposite_run': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.73']])),
                               'Ped_rigth_near': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.50']])),
                               'Ped_rigth_tooNear': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.54']])),
                               'Ped_rigth_tooNear_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.61']])),
                               'Ped_rigth_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.52']])),
                               'Ped_pnear_wave': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.61']])),
                               'Ped_rigth_pnear': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.59']])),
                               'Ped_pnear_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.68']])),
                               'Ped_tooNear_run': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.62']])),
                               'Ped_near_walk': bayes.evaluate_triple(np.array([['Pedestrian', 'Intention', 'Cross_0.50']]))
                            
                               },
                           'evidences_hypothesis_cross':{
                               'NearFromCurb': bayes.evaluate_triple(np.array([['NearFromCurb', 'Action', 'CrossRoad']])),
                               'MiddleDisFromCurb': bayes.evaluate_triple(np.array([['MiddleDisFromCurb', 'Action', 'CrossRoad']])),
                               'FarFromCurb': bayes.evaluate_triple(np.array([['FarFromCurb', 'Action', 'CrossRoad']])),
                               'Stand': bayes.evaluate_triple(np.array([['Stand', 'Action', 'CrossRoad']])),
                               'Walk': bayes.evaluate_triple(np.array([['Walk', 'Action', 'CrossRoad']])),
                               'VehDirection': bayes.evaluate_triple(np.array([['VehDirection', 'Action', 'CrossRoad']])),
                               'LeftDirection': bayes.evaluate_triple(np.array([['LeftDirection', 'Action', 'CrossRoad']])),
                               'OppositeVehDirection': bayes.evaluate_triple(np.array([['OppositeVehDirection', 'Action', 'CrossRoad']])),
                               'RigthDirection': bayes.evaluate_triple(np.array([['RigthDirection', 'Action', 'CrossRoad']])),
                               'Looking': bayes.evaluate_triple(np.array([['Looking', 'Action', 'CrossRoad']])),
                               'NotLooking': bayes.evaluate_triple(np.array([['NotLooking', 'Action', 'CrossRoad']])),
                               'TooNearToEgoVeh': bayes.evaluate_triple(np.array([['TooNearToEgoVeh', 'Action', 'CrossRoad']])),
                                'NearToEgoVeh': bayes.evaluate_triple(np.array([['NearToEgoVeh', 'Action', 'CrossRoad']])),
                               'MiddleDisToEgoVeh': bayes.evaluate_triple(np.array([['MiddleDisToEgoVeh', 'Action', 'CrossRoad']])),
                               'FarToEgoVeh': bayes.evaluate_triple(np.array([['FarToEgoVeh', 'Action', 'CrossRoad']])),
                               'TooFarToEgoVeh': bayes.evaluate_triple(np.array([['TooFarToEgoVeh', 'Action', 'CrossRoad']])),
                               'Ped_wave': bayes.evaluate_triple(np.array([['NoCross_0.62', 'Action', 'CrossRoad']])),
                               'Ped_stand': bayes.evaluate_triple(np.array([['NoCross_0.75', 'Action', 'CrossRoad']])),
                               'Ped_pmoderate': bayes.evaluate_triple(np.array([['NoCross_0.63', 'Action', 'CrossRoad']])),
                               'Ped_rigth': bayes.evaluate_triple(np.array([['NoCross_0.50', 'Action', 'CrossRoad']])),
                               'Ped_vehDir': bayes.evaluate_triple(np.array([['NoCross_0.70', 'Action', 'CrossRoad']])),
                               'Ped_noLook_pfar': bayes.evaluate_triple(np.array([['NoCross_0.69', 'Action', 'CrossRoad']])),
                               'Ped_noLook_rigth': bayes.evaluate_triple(np.array([['NoCross_0.48', 'Action', 'CrossRoad']])),
                               'Ped_noLook_vehDir_walk': bayes.evaluate_triple(np.array([['NoCross_0.69', 'Action', 'CrossRoad']])),
                               'Ped_noLook_vehDir_tooNear': bayes.evaluate_triple(np.array([['NoCross_0.78', 'Action', 'CrossRoad']])),
                               'Ped_noLook_rigth_walk': bayes.evaluate_triple(np.array([['NoCross_0.48', 'Action', 'CrossRoad']])),
                               'Ped_noLook_rigth_near': bayes.evaluate_triple(np.array([['NoCross_0.49', 'Action', 'CrossRoad']])),
                               'Ped_look_opposite': bayes.evaluate_triple(np.array([['NoCross_0.62', 'Action', 'CrossRoad']])),
                               'Ped_look_opposite_near': bayes.evaluate_triple(np.array([['NoCross_0.66', 'Action', 'CrossRoad']])),
                               'Ped_look_near_walk': bayes.evaluate_triple(np.array([['NoCross_0.58', 'Action', 'CrossRoad']])),
                               'Ped_vehDir_tooNear': bayes.evaluate_triple(np.array([['NoCross_0.78', 'Action', 'CrossRoad']])),
                               'Ped_left_run': bayes.evaluate_triple(np.array([['NoCross_0.55', 'Action', 'CrossRoad']])),
                               'Ped_left_near_walk': bayes.evaluate_triple(np.array([['NoCross_0.53', 'Action', 'CrossRoad']])),
                               'Ped_left_near': bayes.evaluate_triple(np.array([['NoCross_0.55', 'Action', 'CrossRoad']])),
                               'Ped_rigth_near_walk': bayes.evaluate_triple(np.array([['NoCross_0.47', 'Action', 'CrossRoad']])),
                               'Ped_rigth_tooNear': bayes.evaluate_triple(np.array([['NoCross_0.55', 'Action', 'CrossRoad']])),
                               'Ped_pnear_stand': bayes.evaluate_triple(np.array([['NoCross_0.65', 'Action', 'CrossRoad']])),
                               'Ped_pmoderate_near': bayes.evaluate_triple(np.array([['NoCross_0.70', 'Action', 'CrossRoad']])),
                               'Ped_pfar_near': bayes.evaluate_triple(np.array([['NoCross_0.80', 'Action', 'CrossRoad']])),
                               'Ped_pnear': bayes.evaluate_triple(np.array([['Cross_0.65', 'Action', 'CrossRoad']])),
                               'Ped_noLook_left_run': bayes.evaluate_triple(np.array([['Cross_0.56', 'Action', 'CrossRoad']])),
                               'Ped_noLook_left_near': bayes.evaluate_triple(np.array([['Cross_0.47', 'Action', 'CrossRoad']])),
                               'Ped_noLook_opposite_walk': bayes.evaluate_triple(np.array([['Cross_0.68', 'Action', 'CrossRoad']])),
                               'Ped_noLook_opposite_tooNear': bayes.evaluate_triple(np.array([['Cross_0.69', 'Action', 'CrossRoad']])),
                               'Ped_noLook_rigth_walk': bayes.evaluate_triple(np.array([['Cross_0.58', 'Action', 'CrossRoad']])),
                               'Ped_noLook_rigth_tooNear': bayes.evaluate_triple(np.array([['Cross_0.58', 'Action', 'CrossRoad']])),
                               'Ped_noLook_rigth_near': bayes.evaluate_triple(np.array([['Cross_0.52', 'Action', 'CrossRoad']])),
                               'Ped_noLook_pmoderate_tooNear': bayes.evaluate_triple(np.array([['Cross_0.50', 'Action', 'CrossRoad']])),
                               'Ped_noLook_near_run': bayes.evaluate_triple(np.array([['Cross_0.54', 'Action', 'CrossRoad']])),
                               'Ped_look_opposite_tooNear': bayes.evaluate_triple(np.array([['Cross_0.49', 'Action', 'CrossRoad']])),
                               'Ped_look_tooNear_walk': bayes.evaluate_triple(np.array([['Cross_0.50', 'Action', 'CrossRoad']])),
                               'Ped_left_near': bayes.evaluate_triple(np.array([['Cross_0.48', 'Action', 'CrossRoad']])),
                               'Ped_left_pnear_near': bayes.evaluate_triple(np.array([['Cross_0.63', 'Action', 'CrossRoad']])),
                               'Ped_left_near_walk': bayes.evaluate_triple(np.array([['Cross_0.48', 'Action', 'CrossRoad']])),
                               'Ped_opposite_run': bayes.evaluate_triple(np.array([['Cross_0.73', 'Action', 'CrossRoad']])),
                               'Ped_rigth_near': bayes.evaluate_triple(np.array([['Cross_0.50', 'Action', 'CrossRoad']])),
                               'Ped_rigth_tooNear': bayes.evaluate_triple(np.array([['Cross_0.54', 'Action', 'CrossRoad']])),
                               'Ped_rigth_tooNear_walk': bayes.evaluate_triple(np.array([['Cross_0.61', 'Action', 'CrossRoad']])),
                               'Ped_rigth_walk': bayes.evaluate_triple(np.array([['Cross_0.52', 'Action', 'CrossRoad']])),
                               'Ped_pnear_wave': bayes.evaluate_triple(np.array([['Cross_0.61', 'Action', 'CrossRoad']])),
                               'Ped_rigth_pnear': bayes.evaluate_triple(np.array([['Cross_0.59', 'Action', 'CrossRoad']])),
                               'Ped_pnear_walk': bayes.evaluate_triple(np.array([['Cross_0.68', 'Action', 'CrossRoad']])),
                               'Ped_tooNear_run': bayes.evaluate_triple(np.array([['Cross_0.62', 'Action', 'CrossRoad']])),
                               'Ped_near_walk': bayes.evaluate_triple(np.array([['Cross_0.50', 'Action', 'CrossRoad']]))
                           },
                           'evidences_hypothesis_ncross':{
                               'NearFromCurb': bayes.evaluate_triple(np.array([['NearFromCurb', 'Action', 'noCrossRoad']])),
                               'MiddleDisFromCurb': bayes.evaluate_triple(np.array([['MiddleDisFromCurb', 'Action', 'noCrossRoad']])),
                               'FarFromCurb': bayes.evaluate_triple(np.array([['FarFromCurb', 'Action', 'noCrossRoad']])),
                               'Stand': bayes.evaluate_triple(np.array([['Stand', 'Action', 'noCrossRoad']])),
                               'Walk': bayes.evaluate_triple(np.array([['Walk', 'Action', 'noCrossRoad']])),
                               'VehDirection': bayes.evaluate_triple(np.array([['VehDirection', 'Action', 'noCrossRoad']])),
                               'LeftDirection': bayes.evaluate_triple(np.array([['LeftDirection', 'Action', 'noCrossRoad']])),
                               'OppositeVehDirection': bayes.evaluate_triple(np.array([['OppositeVehDirection', 'Action', 'noCrossRoad']])),
                               'RigthDirection': bayes.evaluate_triple(np.array([['RigthDirection', 'Action', 'noCrossRoad']])),
                               'Looking': bayes.evaluate_triple(np.array([['Looking', 'Action', 'noCrossRoad']])),
                               'NotLooking': bayes.evaluate_triple(np.array([['NotLooking', 'Action', 'noCrossRoad']])),
                               'TooNearToEgoVeh': bayes.evaluate_triple(np.array([['TooNearToEgoVeh', 'Action', 'noCrossRoad']])),
                               'NearToEgoVeh': bayes.evaluate_triple(np.array([['NearToEgoVeh', 'Action', 'noCrossRoad']])),
                               'MiddleDisToEgoVeh': bayes.evaluate_triple(np.array([['MiddleDisToEgoVeh', 'Action', 'noCrossRoad']])),
                               'FarToEgoVeh': bayes.evaluate_triple(np.array([['FarToEgoVeh', 'Action', 'noCrossRoad']])),
                               'TooFarToEgoVeh': bayes.evaluate_triple(np.array([['TooFarToEgoVeh', 'Action', 'noCrossRoad']])),
                               'Ped_wave': bayes.evaluate_triple(np.array([['NoCross_0.62', 'Action', 'noCrossRoad']])),
                               'Ped_stand': bayes.evaluate_triple(np.array([['NoCross_0.75', 'Action', 'noCrossRoad']])),
                               'Ped_pmoderate': bayes.evaluate_triple(np.array([['NoCross_0.63', 'Action', 'noCrossRoad']])),
                               'Ped_rigth': bayes.evaluate_triple(np.array([['NoCross_0.50', 'Action', 'noCrossRoad']])),
                               'Ped_vehDir': bayes.evaluate_triple(np.array([['NoCross_0.70', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_pfar': bayes.evaluate_triple(np.array([['NoCross_0.69', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_rigth': bayes.evaluate_triple(np.array([['NoCross_0.48', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_vehDir_walk': bayes.evaluate_triple(np.array([['NoCross_0.69', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_vehDir_tooNear': bayes.evaluate_triple(np.array([['NoCross_0.78', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_rigth_walk': bayes.evaluate_triple(np.array([['NoCross_0.48', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_rigth_near': bayes.evaluate_triple(np.array([['NoCross_0.49', 'Action', 'noCrossRoad']])),
                               'Ped_look_opposite': bayes.evaluate_triple(np.array([['NoCross_0.62', 'Action', 'noCrossRoad']])),
                               'Ped_look_opposite_near': bayes.evaluate_triple(np.array([['NoCross_0.66', 'Action', 'noCrossRoad']])),
                               'Ped_look_near_walk': bayes.evaluate_triple(np.array([['NoCross_0.58', 'Action', 'noCrossRoad']])),
                               'Ped_vehDir_tooNear': bayes.evaluate_triple(np.array([['NoCross_0.78', 'Action', 'noCrossRoad']])),
                               'Ped_left_run': bayes.evaluate_triple(np.array([['NoCross_0.55', 'Action', 'noCrossRoad']])),
                               'Ped_left_near_walk': bayes.evaluate_triple(np.array([['NoCross_0.53', 'Action', 'noCrossRoad']])),
                               'Ped_left_near': bayes.evaluate_triple(np.array([['NoCross_0.55', 'Action', 'noCrossRoad']])),
                               'Ped_rigth_near_walk': bayes.evaluate_triple(np.array([['NoCross_0.47', 'Action', 'noCrossRoad']])),
                               'Ped_rigth_tooNear': bayes.evaluate_triple(np.array([['NoCross_0.55', 'Action', 'noCrossRoad']])),
                               'Ped_pnear_stand': bayes.evaluate_triple(np.array([['NoCross_0.65', 'Action', 'noCrossRoad']])),
                               'Ped_pmoderate_near': bayes.evaluate_triple(np.array([['NoCross_0.70', 'Action', 'noCrossRoad']])),
                               'Ped_pfar_near': bayes.evaluate_triple(np.array([['NoCross_0.80', 'Action', 'noCrossRoad']])),
                               'Ped_pnear': bayes.evaluate_triple(np.array([['Cross_0.65', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_left_run': bayes.evaluate_triple(np.array([['Cross_0.56', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_left_near': bayes.evaluate_triple(np.array([['Cross_0.47', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_opposite_walk': bayes.evaluate_triple(np.array([['Cross_0.68', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_opposite_tooNear': bayes.evaluate_triple(np.array([['Cross_0.69', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_rigth_walk': bayes.evaluate_triple(np.array([['Cross_0.58', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_rigth_tooNear': bayes.evaluate_triple(np.array([['Cross_0.58', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_rigth_near': bayes.evaluate_triple(np.array([['Cross_0.52', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_pmoderate_tooNear': bayes.evaluate_triple(np.array([['Cross_0.50', 'Action', 'noCrossRoad']])),
                               'Ped_noLook_near_run': bayes.evaluate_triple(np.array([['Cross_0.54', 'Action', 'noCrossRoad']])),
                               'Ped_look_opposite_tooNear': bayes.evaluate_triple(np.array([['Cross_0.49', 'Action', 'noCrossRoad']])),
                               'Ped_look_tooNear_walk': bayes.evaluate_triple(np.array([['Cross_0.50', 'Action', 'noCrossRoad']])),
                               'Ped_left_near': bayes.evaluate_triple(np.array([['Cross_0.48', 'Action', 'noCrossRoad']])),
                               'Ped_left_pnear_near': bayes.evaluate_triple(np.array([['Cross_0.63', 'Action', 'noCrossRoad']])),
                               'Ped_left_near_walk': bayes.evaluate_triple(np.array([['Cross_0.48', 'Action', 'noCrossRoad']])),
                               'Ped_opposite_run': bayes.evaluate_triple(np.array([['Cross_0.73', 'Action', 'noCrossRoad']])),
                               'Ped_rigth_near': bayes.evaluate_triple(np.array([['Cross_0.50', 'Action', 'noCrossRoad']])),
                               'Ped_rigth_tooNear': bayes.evaluate_triple(np.array([['Cross_0.54', 'Action', 'noCrossRoad']])),
                               'Ped_rigth_tooNear_walk': bayes.evaluate_triple(np.array([['Cross_0.61', 'Action', 'noCrossRoad']])),
                               'Ped_rigth_walk': bayes.evaluate_triple(np.array([['Cross_0.52', 'Action', 'noCrossRoad']])),
                               'Ped_pnear_wave': bayes.evaluate_triple(np.array([['Cross_0.61', 'Action', 'noCrossRoad']])),
                               'Ped_rigth_pnear': bayes.evaluate_triple(np.array([['Cross_0.59', 'Action', 'noCrossRoad']])),
                               'Ped_pnear_walk': bayes.evaluate_triple(np.array([['Cross_0.68', 'Action', 'noCrossRoad']])),
                               'Ped_tooNear_run': bayes.evaluate_triple(np.array([['Cross_0.62', 'Action', 'noCrossRoad']])),
                               'Ped_near_walk': bayes.evaluate_triple(np.array([['Cross_0.50', 'Action', 'noCrossRoad']]))
                           }
                        }
    

    return base_probs