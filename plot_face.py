import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

from utils import *
from dwpose import DWposeDetector

# mouth_x_min = 0.3
# mouth_x_max = 0.7
# mouth_y_min = 0.5
# mouth_y_max = 0.7

# # Filter points that lie within the mouth region
# mouth_points = []
# for point in points_2d:
#     x, y = point
#     if mouth_x_min <= x <= mouth_x_max and mouth_y_min <= y <= mouth_y_max:
#         mouth_points.append(point)

# mouth_points = [mouth_points]

if __name__ == "__main__":
  H = 480
  W = 640

  det_config = './dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
  det_ckpt = './ckpts/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
  pose_config = './dwpose/dwpose_config/dwpose-l_384x288.py'
  pose_ckpt = './ckpts/dw-ll_ucoco_384.pth'
  device = "cuda:0"

  dwprocessor = DWposeDetector(det_config, det_ckpt, pose_config, pose_ckpt, device)

  for img_num in range(1, 9):
    img_path = Path(f"assets/test{img_num}.jpg")
    output_path = Path(f"outputs/test{img_num}.jpg")

    input_image = cv2.imread(img_path)
    H, W, C = input_image.shape

    input_image = HWC3(input_image)
    input_image = resize_image(input_image, resolution=512)
    face_data = dwprocessor(input_image)
    mouth_data = face_data[0]
    
    MOUTH_START_INDEX = 48
    MOUTH_END_INDEX = 69

    # handling reflecting the imae
    x_coords = [i*W for i in mouth_data[:, 0]]
    y_coords = [(1-j)*H for j in mouth_data[:, 1]]

    mouth_x = x_coords[MOUTH_START_INDEX : MOUTH_END_INDEX]
    mouth_y = y_coords[MOUTH_START_INDEX : MOUTH_END_INDEX]

    face_x = x_coords[: MOUTH_START_INDEX]
    face_y = y_coords[: MOUTH_START_INDEX]

    print(f"image: test{img_num}, len(x_coords) = {len(x_coords)}, len(y_coords) = {len(y_coords)}")

    plt.figure(figsize=(8, 8))
    plt.scatter(face_x, face_y, c='gray', marker='o')
    plt.scatter(mouth_x, mouth_y, c='blue', marker='o')

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('2D Face Plot')

    plt.savefig(output_path, format='png', dpi=300)
    print(f"Plot saved as {output_path}")

    plt.close()