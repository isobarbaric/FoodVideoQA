import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from dataclasses import dataclass
from pose.detection.utils import *
from pose.detection.dwpose import PoseDetector

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
POSE_DATA_DIR = DATA_DIR / "pose"
IMG_SOURCE_DIR = POSE_DATA_DIR / "assets"
FACE_PLOT_OUTPUT_DIR = POSE_DATA_DIR / "face-plot"

# constants
MOUTH_START_INDEX = 48
MOUTH_END_INDEX = 69
H = 480
W = 640

LIP_SEPARATION_THRESHOLD = 8

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


def _get_landmarks(pose_detector: PoseDetector, img_path: str) -> FacialLandmarks:
  input_image = cv2.imread(img_path)
  H, W, C = input_image.shape

  input_image = HWC3(input_image)
  input_image = resize_image(input_image, resolution=512)
  
  face_data = pose_detector.get_face(input_image)
  mouth_data = face_data[0]
  
  # handling reflecting the image
  x_coords = [i*W for i in mouth_data[:, 0]]
  y_coords = [(1-j)*H for j in mouth_data[:, 1]]

  return FacialLandmarks(x_coords=x_coords, y_coords=y_coords)


def _is_mouth_open(landmarks: FacialLandmarks) -> bool:
  lip_bottom = np.array(landmarks.lip_bottom_y)
  lip_top = np.array(landmarks.lip_top_y)

  distance = lip_top - lip_bottom
  # print(distance)
  distance = abs(np.mean(distance))
  
  # don't think I need to adjust this since DWPose always seems to output images of the same dimension
  # distance /= H

  return distance > LIP_SEPARATION_THRESHOLD


def determine_mouth_open(pose_detector: PoseDetector, img_path: Path, output_path: Path = None) -> bool:
  landmarks = _get_landmarks(pose_detector, img_path)
  mouth_open = _is_mouth_open(landmarks)

  if output_path is not None:
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
  
  return mouth_open


if __name__ == "__main__":
  pose_detector = PoseDetector()

  if not FACE_PLOT_OUTPUT_DIR.exists():
    FACE_PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

  for img_num in range(1, 5):
    img_path = IMG_SOURCE_DIR / f"test{img_num}.jpg"
    output_path = FACE_PLOT_OUTPUT_DIR / f"test{img_num}.jpg"
    mouth_open = determine_mouth_open(pose_detector, img_path, output_path)
    print(f"test{img_num}: {mouth_open}")