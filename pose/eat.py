from pathlib import Path
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
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
FACE_PLOT_OUTPUT_DIR = DATASET_DIR / "face-plots"

# mouth_open, iou are available for running ablations
def determine_eating(
    generate_bounding_boxes: Callable[[Path], list[BoundingBox]], 
    image_path: Path, 
    bbox_output_path: Path = None,
    face_plot_output_path: Path = None,
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
        raise ValueError("At least one of `mouth_open` and `iou` must be True")

def compute_PRF1(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1

if __name__ == "__main__":
    console = Console()

    model_name = "IDEA-Research/grounding-dino-base"
    generate_bounding_boxes = make_get_bounding_boxes(model_name)

    mouth_open_count = 0
    iou_count = 0
    combined_count = 0
    gt_count = 0

    valid_image_count = 0

    gt_values = []
    predictions_combined = []
    predictions_mouth_only = []
    predictions_iou_only = []

    with open(GROUND_TRUTH_JSON, "r") as f:
        ground_truth = json.load(f)

    for video_dir in sorted(IMAGE_INPUT_DIR.iterdir()):
        if not video_dir.is_dir():
            continue
        
        for img_num in range(1, 10):
            console.print("[green]Processing[/green] ", video_dir.name, f"frame_{img_num}")
            image_path = video_dir / f"frame_{img_num}.jpg"

            output_path = IMAGE_OUTPUT_DIR / video_dir.name / f"frame_{img_num}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            face_plot_output = FACE_PLOT_OUTPUT_DIR / video_dir.name / f"frame_{img_num}.jpg"
            face_plot_output.parent.mkdir(parents=True, exist_ok=True)

            combined = determine_eating(generate_bounding_boxes, image_path, output_path, face_plot_output_path=face_plot_output)
            mouth_only = determine_eating(generate_bounding_boxes, image_path, output_path, include_mouth_open=True, include_iou=False)
            iou_only = determine_eating(generate_bounding_boxes, image_path, output_path, include_mouth_open=False, include_iou=True)

            gt = 1 if ground_truth[video_dir.name][f"frame_{img_num}"] == 1 else 0
            
            gt_values.append(gt)
            predictions_combined.append(combined)
            predictions_mouth_only.append(mouth_only)
            predictions_iou_only.append(iou_only)

            if gt == combined:
                combined_count += 1
            if gt == mouth_only:
                mouth_open_count += 1
            if gt == iou_only:
                iou_count += 1

            valid_image_count += 1
            gt_count += gt

            mouth_only_accuracy = mouth_open_count / valid_image_count
            iou_only_accuracy = iou_count / valid_image_count
            combined_accuracy = combined_count / valid_image_count

            console.print(f"GT: {gt} | Mouth: {mouth_only} | IOU: {iou_only} | Combined: {combined}")
            console.print(f"Accuracy - Mouth only: {mouth_only_accuracy:.2f}", style="bold")
            console.print(f"Accuracy - IOU only: {iou_only_accuracy:.2f}", style="bold")
            console.print(f"Accuracy - Mouth & IOU combined: {combined_accuracy:.2f}", style="bold")

            P_combined, R_combined, F1_combined = compute_PRF1(gt_values, predictions_combined)
            P_mouth, R_mouth, F1_mouth = compute_PRF1(gt_values, predictions_mouth_only)
            P_iou, R_iou, F1_iou = compute_PRF1(gt_values, predictions_iou_only)

            console.print(f"Precision - Mouth only: {P_mouth:.2f}", style="green")
            console.print(f"Recall - Mouth only: {R_mouth:.2f}", style="green")
            console.print(f"F1 - Mouth only: {F1_mouth:.2f}", style="green")

            console.print(f"Precision - IOU only: {P_iou:.2f}", style="blue")
            console.print(f"Recall - IOU only: {R_iou:.2f}", style="blue")
            console.print(f"F1 - IOU only: {F1_iou:.2f}", style="blue")

            console.print(f"Precision - Combined: {P_combined:.2f}", style="bold")
            console.print(f"Recall - Combined: {R_combined:.2f}", style="bold")
            console.print(f"F1 - Combined: {F1_combined:.2f}", style="bold")



    console.print()
    console.print(f"Total images processed: {valid_image_count}")
    console.print(f"Final Results:", style="bold")

    console.print(f"GT: {gt} | Mouth: {mouth_only} | IOU: {iou_only} | Combined: {combined}")
    console.print(f"Accuracy - Mouth only: {mouth_only_accuracy:.2f}", style="bold")
    console.print(f"Accuracy - IOU only: {iou_only_accuracy:.2f}", style="bold")
    console.print(f"Accuracy - Mouth & IOU combined: {combined_accuracy:.2f}", style="bold")

    P_combined, R_combined, F1_combined = compute_PRF1(gt_values, predictions_combined)
    P_mouth, R_mouth, F1_mouth = compute_PRF1(gt_values, predictions_mouth_only)
    P_iou, R_iou, F1_iou = compute_PRF1(gt_values, predictions_iou_only)

    console.print(f"Precision - Mouth only: {P_mouth:.2f}", style="green")
    console.print(f"Recall - Mouth only: {R_mouth:.2f}", style="green")
    console.print(f"F1 - Mouth only: {F1_mouth:.2f}", style="green")

    console.print(f"Precision - IOU only: {P_iou:.2f}", style="blue")
    console.print(f"Recall - IOU only: {R_iou:.2f}", style="blue")
    console.print(f"F1 - IOU only: {F1_iou:.2f}", style="blue")

    console.print(f"Precision - Combined: {P_combined:.2f}", style="bold")
    console.print(f"Recall - Combined: {R_combined:.2f}", style="bold")
    console.print(f"F1 - Combined: {F1_combined:.2f}", style="bold")
