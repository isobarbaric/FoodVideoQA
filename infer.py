import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from utils import *
from dwpose import DWposeDetector

def infer_pose(img_path: Path, output_path: Path):
    input_image = cv2.imread(img_path)

    H, W, C = input_image.shape
    input_image = HWC3(input_image)
    input_image = resize_image(input_image, resolution=512)
    
    detected_map = dwprocessor(input_image)
    detected_map = HWC3(detected_map)
    
    # if not output_path.exists():
    #     output_path.mkdir(parents=True, exist_ok=True)

    # print(f"height: {H}, weight: {W}")

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    
    combined_path = output_path.parent / img_path.name
    output_path = str(combined_path)

    # cv2.imwrite(output_path, detected_map)
    cv2.imwrite(output_path, detected_map)
    

# TODO: set up an argparser for this stuff
if __name__ == '__main__' :
    det_config = './dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
    det_ckpt = './ckpts/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
    pose_config = './dwpose/dwpose_config/dwpose-l_384x288.py'
    pose_ckpt = './ckpts/dw-ll_ucoco_384.pth'
    device = "cuda:0"

    dwprocessor = DWposeDetector(det_config, det_ckpt, pose_config, pose_ckpt, device)

    img_path = Path("assets/test1.jpg")
    output_path = Path("outputs/test1.jpg")
    infer_pose(img_path, output_path)

    # for img_num in range(1, 9):
    #     img_path = Path(f"assets/test{img_num}.jpg")
    #     assert img_path.exists()
    #     output_path = Path(f"outputs/test{img_num}.jpg")
    #     infer_pose(img_path, output_path)
    #     print(f"{img_path} done")