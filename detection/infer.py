import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from utils import *
from dwpose import PoseDetectorConfig, PoseDetector

def infer_pose(pose_detector: PoseDetector, img_path: Path, output_path: Path):
    input_image = cv2.imread(img_path)

    H, W, C = input_image.shape
    input_image = HWC3(input_image)
    input_image = resize_image(input_image, resolution=512)
    
    # TODO: the line below is why this doesn't work
    detected_map = pose_detector(input_image)
    detected_map = HWC3(detected_map)
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    
    combined_path = output_path.parent / img_path.name
    output_path = str(combined_path)

    # cv2.imwrite(output_path, detected_map)
    cv2.imwrite(output_path, detected_map)
 

# TODO: modify __call__ in PoseDetector to return canvas & face data so that infer_pose works
if __name__ == '__main__' :
    img_path = Path("assets/test1.jpg")
    output_path = Path("outputs/test-test1.jpg")

    pose_detector = PoseDetector()
    # infer_pose(pose_detector, img_path, output_path)

    for img_num in range(1, 9):
        img_path = Path(f"assets/test{img_num}.jpg")
        output_path = Path(f"outputs/test-test{img_num}.jpg")
        infer_pose(pose_detector, img_path, output_path)
        print(f"{img_path} done")