from pathlib import Path
import torch
import numpy as np
import cv2

from .yolopv2.yolop_utils import driving_area_mask, lane_line_mask, letterbox


class RoadContextDetector :
    def __init__(self, device, settings, focal_length):
        
        self.device = device
        self.focal_l = focal_length #JAAD Default

        #Load Model
        self.model  = torch.jit.load(settings ["F_YOLOPV2"],map_location=torch.device("cpu"))
        self.model = self.model.to(device)
        self.half = device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half() 
        self.model.eval()
        self.min_m_dis = settings['DISTANCE_CURB_NEAR']
        self.med_m_dis = settings['DISTANCE_CURB_MEDIUM']
        self.resolution = settings['YOLOVP2_RESOLUTION']
        self.scale = settings['SCALE_DETECTION']
    
    def prepare_img (self, img):
        img_size = 640
        stride = 32
        img = letterbox(img, img_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)


        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect_road_context (self, img, orig_img):

        with torch.no_grad(): ## PREVENT ALLOCATION MEMORY
            [pred,anchor_grid],seg,ll= self.model(img)

        self.da_seg_mask = driving_area_mask(self.scale,seg)
        self.ll_seg_mask = lane_line_mask(self.scale,ll)

        self.x_shape,_ = self.ll_seg_mask.shape
        #self.paint_seg_mask(orig_img)

        return self.da_seg_mask, self.ll_seg_mask
    
    def find_seg_mask (self, left_move, right_move, pedX, pedY, value):
        
        if left_move <= 0 or left_move >= self.resolution:
            left_move = pedX
        if right_move >= self.resolution or right_move <= 0:
            right_move = pedX

        if self.x_shape == pedY:
            pedY = pedY - 1
            print("entre")

        if self.ll_seg_mask[pedY][left_move] == 1 or self.da_seg_mask[pedY][left_move] == 1:
            return value
        elif self.ll_seg_mask[pedY][right_move] == 1 or self.da_seg_mask[pedY][right_move] == 1:
            return value
        
        return 2
    
    def paint_seg_mask(self, img):
        #img = np.transpose(img, (1, 2, 0))

        da_mask = self.da_seg_mask > 0
        ll_mask = self.ll_seg_mask > 0

        # Paint
        img[da_mask] = (0, 255, 0)   # driving area
        img[ll_mask] = (0, 0, 255)   # lane lines

        # Show
        cv2.imshow("Painted Image", img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()


    def pedestrian_near_road(self, pedbox):

        #https://blog.roboflow.com/computer-vision-measure-distance/
        # m = pixel/ pixels ratio ---- pixels ratio = pixels size / real value m
        W, H = self.get_known_width_height(0) #Known width
        pixels_ratio_x = round((pedbox[2]-pedbox[0])/ W ,2)
        pixels_ratio_y = round((pedbox[3]-pedbox[1])/ H ,2)
        pixels_ratio = min(pixels_ratio_x,pixels_ratio_y)
        pixel_distance = self.min_m_dis * pixels_ratio
        # pixel_distance = distancia real cerca (m) * pixels_ratio
        # buscar por pixel distance a la derecha y a la izquierda



        pedX = int((pedbox[0]+pedbox[2])/2)
        left_move = int(pedX - pixel_distance)
        rigth_move = int(pedX + pixel_distance)

        value = self.find_seg_mask(left_move,rigth_move, pedX, int(pedbox[3]), 0)
        if value == 2:
            pixel_distance = self.med_m_dis * pixels_ratio
            left_move = int(pedX - pixel_distance)
            rigth_move = int(pedX + pixel_distance)
            return self.find_seg_mask(left_move,rigth_move, pedX, int (pedbox[3]), 1)
        
        return value

    def get_proximity(self, proximity):
        if proximity == 0:
            return 'NearFromCurb'
        elif proximity == 1:
            return 'MiddleDisFromCurb'
        return 'FarFromCurb'


    def get_pedestrian_lat_lon(self, x0,xf,y0,yf):
        # CAMERA AND OBJECT PARAMETERS
        W = 0.6
        H = 1.70

        #Focal length in pixels
        fx = 2143
        fy = 2143

        #Camera center in pixels
        cx =  140#460#960#965
        cy =  300#207
    
        du = xf - x0
        dv = yf - y0
        xc = (x0 + xf) / 2.0
    
        if du == 0:
            du = 1
        
        if dv == 0:
            dv = 1
        
    
        Z1 = W * fx / du
        Z2 = H * fy / dv

        Z = min(Z1, Z2)
        X = Z / fx * (xc - cx)
    
        return Z, X
    
    def get_focal_length(self,settings, dataset):
        if dataset == settings["JAAD_NAME"]:
            self.focal_l = (129 * 1.7)
        else:
            self.focal_l = (81 * 1.9)

    def get_pedestrian_distance(self,x0,xf,y0,yf,clss):
        # https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

        # Find focal length F = P (object width in pixels) * D (Distance from camera) / W (Known width)
        W, H = self.get_known_width_height(clss) #Known width
        F = self.focal_l /W #2143 129 * 1.7 for JAAD
        # get distance
        
        # Pixels
        px = xf - x0
        py = yf - y0
        # D = (W x F)/P
        if px != 0 and py != 0:
            dx = (W*F)/px
            dy = (H*F)/py
            dis = min(dx,dy)
            return round(dis, 2)
        else:
            return 0.0
    
    def get_known_width_height(self, clss):
        
        if clss == 0: #Adult
            return 0.6 , 1.70 #https://www.dimensions.com/element/two-lane-corridor
        elif clss == 1: #Child
            return 0.6 , 1.19 #https://www.dimensions.com/element/6-year-old-children-kids
        elif clss == 2: #crutches
            return 0.8, 1.70
        elif clss == 3: #push wheelchair
            return  1.06, 1.70
        elif clss == 4: #walking frame
            return 0.65, 1.70
        return 1.06, 0.91 #WheelChair Standard size
        
        
def traduce_veh_action(action):
    if len(action) > 0:
        action = action[0].attrib['action']
        if action == 'moving_fast':
            return 0
        if action == 'accelerating':
            return 1
        if action == 'decelerating':
            return 2
        if action == 'moving_slow':
            return 3
        if action == 'stopped':
            return 4
    return 0

def traduce_zebra_crossing(zebra):
    if len(zebra) > 0:
        zebra = zebra[0].attrib['ped_crossing']
        return zebra
    return 0

def traduce_zebra_crossing_pie(zebra):
    if zebra is not None:
        zebra = zebra.attrib['signalized']
        if zebra == 'C' or zebra == 'CS':
            return 1
    return 0




    
