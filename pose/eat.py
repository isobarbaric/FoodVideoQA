from pathlib import Path
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from pose.localization.bbox import BoundingBox
from typing import Callable
from rich.console import Console
from pose.localization.dino import make_get_bounding_boxes, determine_iou
from pose.detection.face_plotting import determine_mouth_open, determine_mouth_bounding_box
from pose.detection.pose_detector import PoseDetector
import json

ROOT_DIR = Path(__file__).parent.parent

DATASET_DIR = ROOT_DIR / "data_final"

LLM_DATA_DIR = DATASET_DIR / "llm"
IMAGE_INPUT_DIR = LLM_DATA_DIR / "frames"
DATA_JSON = DATASET_DIR / "data.json"

POSE_DATA_DIR = DATASET_DIR / "pose"
IMAGE_OUTPUT_DIR = POSE_DATA_DIR / "annotated_frames"
FACE_PLOT_OUTPUT_DIR = POSE_DATA_DIR / "face_plots"

# set to False for inference, set to True if you want to validate with (nutriquest) ground truth data
ANALYZE_WITH_GT = False
GROUND_TRUTH_JSON = DATASET_DIR / "nutriquest.json"

# mouth_open, iou are available for running ablations
def determine_eating(
    generate_bounding_boxes: Callable[[Path], list[BoundingBox]], 
    image_path: Path, 
    bbox_output_path: Path = None,
    face_plot_output_path: Path = None,
    mouth_bbox_output_path: Path = None,
    include_mouth_open: bool = True,
    include_iou: bool = True
):
    pose_detector = PoseDetector()

    mouth_open = determine_mouth_open(pose_detector, image_path, face_plot_output_path)
    (x_min, y_min, x_max, y_max) = determine_mouth_bounding_box(pose_detector, image_path, mouth_bbox_output_path)
    iou_condition, _ = determine_iou((x_min, y_min, x_max, y_max), generate_bounding_boxes, image_path, bbox_output_path)

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

    if ANALYZE_WITH_GT:
        with open(GROUND_TRUTH_JSON, "r") as f:
            ground_truth = json.load(f)
    
    with open(DATA_JSON, "r") as file:
        data = json.load(file)
    

    for video_index, video_dir in enumerate(sorted(IMAGE_INPUT_DIR.iterdir())):
        if not video_dir.is_dir():
            continue
    
        video_name = f"video_{video_index + 1}.mp4"
        video_data = next((v for v in data if v["video_name"] == video_name), None)

        if not video_data:
            console.print(f"[red]Video {video_name} not found in data.json[/red]")
            continue

        for frame_data in video_data["frames"]:
            frame_number = frame_data["frame_number"]
            frame_name = f"frame{frame_number}"

            console.print(f"[green]Processing[/green] {video_name} {frame_name}")

            image_path = video_dir / f"{frame_name}.jpg"
            output_path = IMAGE_OUTPUT_DIR / video_name / f"{frame_name}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            face_plot_output = FACE_PLOT_OUTPUT_DIR / video_name / f"{frame_name}.jpg"
            face_plot_output.parent.mkdir(parents=True, exist_ok=True)

            combined = determine_eating(generate_bounding_boxes, image_path, output_path, face_plot_output_path=face_plot_output, mouth_bbox_output_path=None)
            mouth_only = determine_eating(generate_bounding_boxes, image_path, output_path, include_mouth_open=True, include_iou=False)
            iou_only = determine_eating(generate_bounding_boxes, image_path, output_path, include_mouth_open=False, include_iou=True)

            frame_data["eating"] = 1 if combined else 0
            print("eating: ", combined)

            if ANALYZE_WITH_GT:
                img_num = frame_name[5:]
                gt = 1 if ground_truth[video_dir.name][f"frame{img_num}"] == 1 else 0
                
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

    with open(DATA_JSON, "w") as file:
        json.dump(data, file, indent=4)

    if ANALYZE_WITH_GT:
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
