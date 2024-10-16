from pathlib import Path
from pose.localization.bbox import BoundingBox
from typing import Callable
from rich.console import Console
from pose.localization.dino import make_get_bounding_boxes, determine_iou
from pose.detection.face_plotting import determine_mouth_open
from pose.detection.pose_detector import PoseDetector
import json

ROOT_DIR = Path(__file__).parent.parent

DATASET_DIR = ROOT_DIR / "dataset"
IMAGE_INPUT_DIR = DATASET_DIR / "frames"
IMAGE_OUTPUT_DIR = DATASET_DIR / "annotated-frames"
GROUND_TRUTH_JSON = DATASET_DIR / "ground_truth.json"

DATA_DIR = ROOT_DIR / "data"
LOCALIZATION_DIR = DATA_DIR / "localization"
FACE_PLOT_OUTPUT_DIR = LOCALIZATION_DIR / "face-plots"

# mouth_open, iou are available for running eblations
def determine_eating(
    generate_bounding_boxes: Callable[[Path], list[BoundingBox]], 
    image_path: Path, 
    bbox_output_path: Path = None,
    face_plot_output_path: Path = FACE_PLOT_OUTPUT_DIR,
    include_mouth_open: bool = True,
    include_iou: bool = True
):
    pose_detector = PoseDetector()

    mouth_open = determine_mouth_open(pose_detector, image_path, face_plot_output_path)
    iou_condition, _ = determine_iou(generate_bounding_boxes, image_path, bbox_output_path)

    if include_mouth_open and include_iou:
        return mouth_open and iou_condition
    elif include_mouth_open:
        return mouth_open
    elif include_iou:
        return iou_condition
    else:
        raise ValueError("atleast one of `mouth_open` and `iou` must be True")


# TODO: need to integrate ability to read in ground truth values (maybe make a dictionary here?)
if __name__ == "__main__":
    console = Console()

    model_name = "IDEA-Research/grounding-dino-base"
    generate_bounding_boxes = make_get_bounding_boxes(model_name)

    mouth_open_cnt = 0
    iou_cnt = 0
    combined_cnt = 0

    valid_image_cnt = 0

    with open(GROUND_TRUTH_JSON, "r") as f:
        ground_truth = json.load(f)

    # iterate through all video directories inside IMAGE_INPUT_DIR
    for video_dir in sorted(IMAGE_INPUT_DIR.iterdir()):
        if not video_dir.is_dir():
            continue
        
        for img_num in range(1, 10):
            console.print("[green]Processing[/green] ", video_dir.name, f"/frame_{img_num}")
            image_path = video_dir / f"frame_{img_num}.jpg"
            output_path = IMAGE_OUTPUT_DIR / video_dir.name / f"frame_{img_num}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            combined, _ = determine_eating(generate_bounding_boxes, image_path, output_path)
            mouth_only = determine_eating(generate_bounding_boxes, image_path, output_path, include_mouth_open=True, include_iou=False)
            iou_only = determine_eating(generate_bounding_boxes, image_path, output_path, include_mouth_open=False, include_iou=True)

            gt = True if ground_truth[video_dir.name][f"frame_{img_num}"] == 1 else False

            if gt == combined:
                combined_cnt += 1
            if gt == mouth_only:
                mouth_open_cnt += 1
            if gt == iou_only:
                iou_cnt += 1
            
            console.print(f"GT: {gt} | Mouth: {mouth_only} | IOU: {iou_only} | Combined: {combined}")
            console.print()

            valid_image_cnt += 1

    mouth_only_accuracy = mouth_open_cnt / valid_image_cnt
    iou_only_accuracy = iou_cnt / valid_image_cnt
    combined_accuracy = combined_cnt / valid_image_cnt

    console.print(f"Mouth (only): {mouth_only_accuracy}")
    console.print(f"IOU (only): {iou_only_accuracy}")
    console.print(f"Mouth & IOU: {combined_accuracy}")
