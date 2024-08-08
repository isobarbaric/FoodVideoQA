import cv2
from pathlib import Path
from pose.detection.inference_utils import HWC3, resize_image
from pose.detection.pose_detector import PoseDetector

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
POSE_DATA_DIR = DATA_DIR / "pose"
IMG_SOURCE_DIR = POSE_DATA_DIR / "assets"
INFERENCE_OUTPUT_DIR = POSE_DATA_DIR / "inference"


def infer_pose(pose_detector: PoseDetector, img_path: Path, output_path: Path):
    if not img_path.exists():
        raise ValueError(f"No image exists at image path {img_path}")

    input_image = cv2.imread(img_path)

    H, W, C = input_image.shape
    input_image = HWC3(input_image)
    input_image = resize_image(input_image, resolution=512)
    
    # turned on infer flag to create a plot instead of returning landmarks
    # TODO: make "infer" flag something more intuitive
    detected_map = pose_detector.infer(input_image)
    detected_map = HWC3(detected_map)
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    
    combined_path = output_path.parent / img_path.name
    output_path = str(combined_path)

    cv2.imwrite(output_path, detected_map)
 

# TODO: modify __call__ in PoseDetector to return canvas & face data so that infer_pose works
if __name__ == '__main__' :
    pose_detector = PoseDetector()

    if not INFERENCE_OUTPUT_DIR.exists():
        INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    for img_num in range(1, 5):
        img_path = IMG_SOURCE_DIR / f"test{img_num}.jpg"
        output_path = INFERENCE_OUTPUT_DIR / f"test{img_num}.jpg"
        infer_pose(pose_detector, img_path, output_path)
        print(f"{img_path} done")
