
import torch
import torch.nn as nn
import numpy as np

from .action_recog.action_model_config import ModelConfig, make_model

class ActionRecognizer ():

    def __init__(self, config, device):

        dataset = config['DATASET']
        n_heads = config['N_HEADS']
        n_layers = config['N_LAYERS']
        embed_dim = config['EMBED_DIM']
        dropout = config['DROPOUT']
        n_encoder = config['N_ENCODER']
        mlp_head_size = config['MLP']
        d_model = 64 * n_heads
        d_ff = d_model * 4
        embed_dim = config['EMBED_DIM']
        self.n_frames = config[dataset]['FRAMES']
        classes = config[dataset]['CLASSES']
        keypoints = config[dataset]['KEYPOINTS']*2
        self.device = device
        configuration = ModelConfig(n_heads=n_heads, n_layers=n_layers,d_model=d_model, 
        d_ff=d_ff, dropout= dropout, n_encoder=n_encoder, embed_dim=embed_dim,
        n_frames=self.n_frames, n_keypoints=keypoints, mlp_size=mlp_head_size, device=device)
        self.action_recog = make_model(configuration, device, classes)
        self.action_recog.load_state_dict(torch.load('src/cross_predictor/cross_predictor/features_extractor/weigths/action.ckpt',
                     map_location=device ))
        self.action_recog.eval()
        self.labels = config['ACT_ACTION_LABELS']
        self.buffer_window = []
        self.softmax = nn.Softmax(dim=1)
        self.colors = [[11, 201, 83], [0, 124,255], [182, 255, 2], [255, 0, 0], [62, 3, 2],[0, 0, 0]]
        self.buffer_skeletons = {}
    

    def detect_action(self, skeleton,skeleton_3d):

        features_keypoint = []
        pred_index = -1
        label = 'OBSERVING'
        value = 4
        color = self.colors[5]
        for i, row in enumerate(skeleton):
            features_keypoint.append(row[0])
            features_keypoint.append(row[1])

        if len(skeleton_3d) == self.n_frames :
            # convert input to tensor
            keypoints = torch.Tensor(np.array(skeleton_3d, dtype=np.float32)).to(self.device)

            # add extra dimension
            keypoints = torch.unsqueeze(keypoints, dim=0)
            with torch.no_grad(): ## PREVENT ALLOCATION MEMORY
                y_pred = self.action_recog(keypoints)
            prob = self.softmax(y_pred)
            top_p, top_class = prob.topk(1, dim = 1)
            pred_index = prob.argmax(1)
            value = pred_index.cpu().detach().numpy()
            value = value[0]
            label = self.labels[value]
            probability = top_p.cpu().detach().numpy()[0][0]
            #print("ACTION: "+value+" - "+str(probability))
            color = self.colors[int(value)]
        


        return self.get_action(value)
    
    def get_action(self, action):
        if action == 0 or action == 2:    #Stand and Wave
            return 'Stand'
        elif action == 1 or action == 3: #Walk and Run
            return 'Walk'
        return 'Na'


    def save_buffer_skeleton(self, id_person, skeleton):
        buffer = self.buffer_skeletons.get(id_person)
        
        if buffer is None:
            buffer = [self.create_array_skeleton(skeleton)]
        else :
            if len(buffer) < 30 :
                buffer.append(self.create_array_skeleton(skeleton))
            elif len(buffer) == 30:
                buffer = buffer[1:]
                buffer.append(self.create_array_skeleton(skeleton))
            

        self.buffer_skeletons [id_person] = buffer

        return buffer
    
    def create_array_skeleton (self, skeleton):
        key_norm=[1,2,7,8]
        pose = skeleton.copy()
        features_keypoint = []
        row = self.scale_and_center_keypoints(pose,key_norm[0],key_norm[1],key_norm[2],key_norm[3])
        #row = scale_to_unit(row)
        values = []
        for value in row:
            values.append(value[0])
            values.append(value[1])
        skeleton_new = features_keypoint.append(values)
        #return np.array(features_keypoint)
        return np.array(values)
    
    def scale_and_center_keypoints(self,pose, center_1, center_2, m_k_1, m_k_2):
        zero_point = (pose[center_1, :2] + pose[center_2,:2]) / 2
        module_keypoint = (pose[m_k_1, :2] + pose[m_k_2,:2]) / 2
        scale_mag = np.linalg.norm(zero_point - module_keypoint)
        if scale_mag < 1:
            scale_mag = 1
        pose[:,:2] = (pose[:,:2] - zero_point) / scale_mag
        return pose