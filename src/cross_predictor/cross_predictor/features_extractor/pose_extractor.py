import numpy as np
import torch
import cv2
import math
import logging
from .pedrec_net.pedrec_net_config import PedRecNet50Config, PedRecNetConfig
from .pedrec_net.PedRecNet import PedRecNet
from .utils.time_helper import timed
from .utils.bb_helper import bb_to_center_scale
from .utils.augmentation_helper import get_affine_transforms, affine_transform_coords_2d
from typing import Union, List, Tuple, Dict
from torchvision import transforms

class PoseExtractor:
    
    def __init__(self):
        
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pose_model = PedRecNet50Config()
        self.pose_recognizer = self.init_pose_model(PedRecNet(self.pose_model))
        self.pose_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                    ])


    def init_pose_model(self, pose_model: Union[torch.nn.Module]):
        pose_model.load_state_dict(torch.load( 'src/cross_predictor/cross_predictor/features_extractor/weigths/pose.pth'))
        model_parameters = filter(lambda p: p.requires_grad, pose_model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(
            "Loaded PoseHRNet (weights: {}). Num of trainable params: {}".format( 'weigths/pose.pth', num_params))
        pose_model = pose_model.to(self.device)
        pose_model.eval()
        return pose_model
    
    def extract_pose(self, image, box):
        pose_time, pose_preds = timed(lambda: self.pedrec_recognizer(self.pose_recognizer, self.pose_model, image, [box], self.device))
        orientation = pose_preds['orientations'][0]
        skeleton = pose_preds['skeletons'][0]
        return self.get_orientation_linguistic(self.get_body_orientation(math.degrees(orientation[0][1]))), skeleton
    
    def get_orientation_linguistic(self, orientations):
        if orientations >=0 and orientations < 90:
            return 'VehDirection'
        elif orientations >= 90 and orientations <= 180:
            return 'LeftDirection'
        elif orientations > 180 and orientations < 270:
            return 'OppositeVehDirection'
        return 'RigthDirection'
    
    def pedrec_recognizer(self, pose_model: Union[torch.nn.Module],
                      cfg: PedRecNetConfig,
                      img: np.ndarray, bbs: List[np.ndarray], device) -> Dict[str, np.ndarray]:
        centers = []
        scales = []
        for bb in bbs:
            center, scale = bb_to_center_scale(bb, cfg.model.input_size)
            centers.append(center)
            scales.append(scale)
        pose2d_preds = []
        pose3d_preds = []
        orientation_preds = []
        if len(centers) > 0:
            pose2d_preds, pose3d_preds, orientation_preds, model_input_bbs = self.get_pose_estimation_prediction(pose_model, cfg, img, centers, scales,
                                                                                    transform=self.pose_transform, device=device)
        return {"skeletons": pose2d_preds,
                "skeletons_3d": pose3d_preds,
                "orientations": orientation_preds,
                "bbs": bbs}
    
    def get_pose_estimation_prediction(self, pose_model: torch.nn.Module,
                                   cfg: PedRecNetConfig,
                                   img: np.ndarray,
                                   centers: List[np.ndarray],
                                   scales: List[np.ndarray],
                                   transform,
                                   device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        rotation = 0

        # pose estimation transformation
        model_inputs = []
        trans_invs = []

        # PoseResnet pretrain was trained on BGR, thus keep it for now
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for center, scale in zip(centers, scales):
            trans, trans_inv = get_affine_transforms(center, scale, rotation, cfg.model.input_size, add_inv = True)
            trans_invs.append(trans_inv)
            # Crop smaller image of people
            model_input = cv2.warpAffine(
                img,
                trans,
                (int(cfg.model.input_size.width), int(cfg.model.input_size.height)),
                flags=cv2.INTER_LINEAR)
            model_input = transform(model_input)
            model_inputs.append(model_input)
        trans_invs = torch.tensor(np.array(trans_invs), dtype=torch.float32).to(device)
        model_inputs_torch = torch.stack(model_inputs)
        with torch.no_grad(): ## PREVENT ALLOCATION MEMORY
            output = pose_model(model_inputs_torch.to(device))

        pose_coords_2d = output[0]
        pose_coords_2d[:, :, 0] *= cfg.model.input_size.width
        pose_coords_2d[:, :, 1] *= cfg.model.input_size.height
        for i in range(pose_coords_2d.shape[0]):
            pose_coords_2d[i, :, :2] = affine_transform_coords_2d(pose_coords_2d[i], trans_invs[i], pose_coords_2d.device)

        pose_coords_2d = pose_coords_2d[:,:17,:] #Reduce for action recognition 24 openpose
        pose_2d_pred = pose_coords_2d.cpu().detach().numpy()
        pose_3d_pred = output[1].cpu().detach().numpy()
        orientation_pred = output[2].cpu().detach().numpy()

        # return normalized values as well as the denormalized ones... 

        # denormaliz
        orientation_pred[:, :, 0] *= math.pi
        orientation_pred[:, :, 1] *= 2 * math.pi
        pose_3d_pred[:, :, :3] *= 3000
        pose_3d_pred[:, :, :3] -= 1500

        return pose_2d_pred, pose_3d_pred, orientation_pred, model_inputs
    
    def get_body_orientation(self, phi_degree):
        if phi_degree >= 0.0 and phi_degree < 45.0 :
            phi_degree = phi_degree + 360
        new_degree = phi_degree - 45
        
        return round(new_degree,2)
    
