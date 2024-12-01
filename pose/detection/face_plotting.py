import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from dataclasses import dataclass
from pose.detection.inference_utils import HWC3, resize_image
from pose.detection.pose_detector import PoseDetector
from hyperparameters import LIP_SEPARATION_THRESHOLD
import pose.detection.drawing_utils as drawing_utils

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "dataset_old"
FRAME_DIR = DATA_DIR / "frames"
IMG_SOURCE_DIR = FRAME_DIR / "video_1"
FACE_PLOT_OUTPUT_DIR = DATA_DIR / "mouth-face-plot" / "video_1"
BOUNDING_BOX_OUTPUT_DIR = DATA_DIR / "mouth-bounding-box" / "video_1"

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


def _get_landmarks(pose_detector: PoseDetector, img_path: str) -> FacialLandmarks:
  input_image = cv2.imread(img_path)
  H, W, C = input_image.shape

  input_image = HWC3(input_image)
  input_image = resize_image(input_image, resolution=512)
  
  data = pose_detector.extract_data(input_image)
  face_data = data.faces

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

  return distance >= LIP_SEPARATION_THRESHOLD

def determine_mouth_bounding_box(pose_detector: PoseDetector, img_path: Path, output_path: Path = None) -> tuple[int]:
  landmarks = _get_landmarks(pose_detector, img_path)
  
  if not img_path.exists():
    raise ValueError(f"No image found at image path {img_path}")

  # draws the exact points of the mouth
  img = cv2.imread(str(img_path))
  H, W, C = img.shape
  mouth_lmks = [((x / W), 1 - (y / H)) for x, y in zip(landmarks.mouth_x, landmarks.mouth_y)]
  # img = drawing_utils.draw_facepose(img, [mouth_lmks])

  # Convert normalized coordinates back to pixel values for bounding box calculation
  pixel_mouth_lmks = [(int(x * W), int(y * H)) for x, y in mouth_lmks]
  x_min = min([x for x, y in pixel_mouth_lmks])
  x_max = max([x for x, y in pixel_mouth_lmks])
  y_min = min([y for x, y in pixel_mouth_lmks])
  y_max = max([y for x, y in pixel_mouth_lmks])
  img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

  if output_path is not None:
    cv2.imwrite(str(output_path), img)
  
  # return bbox coordinates
  return (x_min, y_min, x_max, y_max)


def determine_mouth_open(pose_detector: PoseDetector, img_path: Path, output_path: Path = None) -> bool:
  if not img_path.exists():
    raise ValueError(f"No image found at image path {img_path}")

  landmarks = _get_landmarks(pose_detector, img_path)
  mouth_open = _is_mouth_open(landmarks)

  if output_path is not None:
    if not output_path.parent.exists():
      output_path.parent.mkdir(parents=True, exist_ok=False)
      
    plt.figure(figsize=(8, 8))
    plt.scatter(landmarks.face_x, landmarks.face_y, c='gray', marker='o', s=100)
    plt.scatter(landmarks.mouth_x, landmarks.mouth_y, c='blue', marker='o', s=100)
    plt.scatter(landmarks.lip_top_x, landmarks.lip_top_y, c='red', marker='o', s=100)
    plt.scatter(landmarks.lip_bottom_x, landmarks.lip_bottom_y, c='green', marker='o', s=100)

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('off')

    plt.savefig(output_path, format='png', dpi=300)
    # print(f"Plot saved as {output_path}")
    plt.close()
  
  return mouth_open


if __name__ == "__main__":
  pose_detector = PoseDetector()

  if not BOUNDING_BOX_OUTPUT_DIR.exists():
    BOUNDING_BOX_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

  for img_num in range(1, 10):
    img_path = IMG_SOURCE_DIR / f"frame_{img_num}.jpg"
    output_path = BOUNDING_BOX_OUTPUT_DIR / f"frame_{img_num}.jpg"
    img = determine_mouth_bounding_box(_get_landmarks(pose_detector, img_path), img_path, output_path)
    print(f"Bounding box drawn for frame_{img_num}")


  # if not FACE_PLOT_OUTPUT_DIR.exists():
  #   FACE_PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
  
  # for img_num in range(1, 10):
  #   img_path = IMG_SOURCE_DIR / f"frame_{img_num}.jpg"
  #   output_path = FACE_PLOT_OUTPUT_DIR / f"frame_{img_num}.jpg"
  #   mouth_open = determine_mouth_open(pose_detector, img_path, output_path)
  #   print(f"frame_{img_num}: {mouth_open}")