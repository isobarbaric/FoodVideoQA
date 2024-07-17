import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from dataclasses import dataclass

from utils import *
from dwpose import DWposeDetector

# constants
MOUTH_START_INDEX = 48
MOUTH_END_INDEX = 69
H = 480
W = 640


@dataclass
class FacialLandmarks:
  x_coords: list[int]
  y_coords: list[int]

  LIP_TOP_COORDS: list[int] = (61, 62, 63)
  LIP_BOTTOM_COORDS: list[int] = (65, 66, 67)

  @property
  def face_x(self) -> list[int]:
    return self.x_coords[:MOUTH_START_INDEX]

  @property
  def face_y(self) -> list[int]:
    return self.y_coords[:MOUTH_START_INDEX]

  @property
  def mouth_x(self) -> list[int]:
    return self.x_coords[MOUTH_START_INDEX:MOUTH_END_INDEX]

  @property
  def mouth_y(self) -> list[int]:
    return self.y_coords[MOUTH_START_INDEX:MOUTH_END_INDEX]

  @property
  def lip_top_x(self) -> list[int]:
    return [self.x_coords[num] for num in self.LIP_TOP_COORDS]

  @property
  def lip_top_y(self) -> list[int]:
    return [self.y_coords[num] for num in self.LIP_TOP_COORDS]

  @property
  def lip_bottom_x(self) -> list[int]:
    return [self.x_coords[num] for num in self.LIP_BOTTOM_COORDS]

  @property
  def lip_bottom_y(self) -> list[int]:
    return [self.y_coords[num] for num in self.LIP_BOTTOM_COORDS]


def get_landmarks(img_path) -> FacialLandmarks:
  input_image = cv2.imread(img_path)
  H, W, C = input_image.shape

  input_image = HWC3(input_image)
  input_image = resize_image(input_image, resolution=512)
  face_data = dwprocessor(input_image)
  mouth_data = face_data[0]
  
  # handling reflecting the imae
  x_coords = [i*W for i in mouth_data[:, 0]]
  y_coords = [(1-j)*H for j in mouth_data[:, 1]]

  return FacialLandmarks(x_coords=x_coords, y_coords=y_coords)

  # mouth_x = x_coords[MOUTH_START_INDEX : MOUTH_END_INDEX]
  # mouth_y = y_coords[MOUTH_START_INDEX : MOUTH_END_INDEX]

  # face_x = x_coords[: MOUTH_START_INDEX]
  # face_y = y_coords[: MOUTH_START_INDEX]

  # # lip_top_coords = [49, 50, 51, 52, 53, 54, 55]
  # # lip_bottom_coords = [56, 57, 58, 59, 60]

  # lip_top_coords = [61, 62, 63]
  # lip_bottom_coords = [65, 66, 67]

  # lip_top_x = [x_coords[num] for num in lip_top_coords]
  # lip_top_y = [y_coords[num] for num in lip_top_coords]

  # lip_bottom_x = [x_coords[num] for num in lip_bottom_coords]
  # lip_bottom_y = [y_coords[num] for num in lip_bottom_coords]

  # return x_coords, y_coords, face_x, face_y, mouth_x, mouth_y, lip_top_x, lip_top_y, lip_bottom_x, lip_bottom_y


if __name__ == "__main__":
  det_config = './dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
  det_ckpt = './ckpts/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
  pose_config = './dwpose/dwpose_config/dwpose-l_384x288.py'
  pose_ckpt = './ckpts/dw-ll_ucoco_384.pth'
  device = "cuda:0"

  dwprocessor = DWposeDetector(det_config, det_ckpt, pose_config, pose_ckpt, device)

  for img_num in range(1, 9):
    img_path = Path(f"assets/test{img_num}.jpg")
    output_path = Path(f"outputs/test{img_num}_mouth.jpg")

    landmarks = get_landmarks(img_path)
    print(f"image: test{img_num}, len(x_coords) = {len(landmarks.x_coords)}, len(y_coords) = {len(landmarks.y_coords)}")

    plt.figure(figsize=(8, 8))
    plt.scatter(landmarks.face_x, landmarks.face_y, c='gray', marker='o')
    plt.scatter(landmarks.mouth_x, landmarks.mouth_y, c='blue', marker='o')
    plt.scatter(landmarks.lip_top_x, landmarks.lip_top_y, c='red', marker='o')
    plt.scatter(landmarks.lip_bottom_x, landmarks.lip_bottom_y, c='green', marker='o')

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('2D Face Plot')

    plt.savefig(output_path, format='png', dpi=300)
    print(f"Plot saved as {output_path}")

    plt.close()