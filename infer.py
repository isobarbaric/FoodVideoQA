import os
import cv2
import glob
import numpy as np
from PIL import Image

from utils import *
from dwpose import DWposeDetector


if __name__ == '__main__' :
    image_dirs = ["./assets/test.jpeg", "./assets/test1.jpeg", "./assets/test2.jpeg", "./assets/test3.jpeg"]
    
    det_config = './dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
    det_ckpt = './ckpts/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
    pose_config = './dwpose/dwpose_config/dwpose-l_384x288.py'
    pose_ckpt = './ckpts/dw-ll_ucoco_384.pth'
    
    device = "cuda:0"
    dwprocessor = DWposeDetector(det_config, det_ckpt, pose_config, pose_ckpt, device)
    
    annotator_ckpts_path = "./ckpts"

    save_dir = "./outputs"
    os.makedirs(save_dir, exist_ok=True)

    for image_dir in image_dirs:
        input_image = cv2.imread(image_dir)
        H, W, C = input_image.shape
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, resolution=512)
        
        detected_map = dwprocessor(input_image)
        detected_map = HWC3(detected_map)
        
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(save_dir, image_dir.split('/')[-1]), detected_map)