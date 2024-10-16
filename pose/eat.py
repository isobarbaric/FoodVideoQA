from pathlib import Path
from pose.localization.bbox import BoundingBox
from typing import Callable
from rich.console import Console
from pose.localization.dino import make_get_bounding_boxes, determine_iou
from pose.detection.face_plotting import determine_mouth_open
from pose.detection.pose_detector import PoseDetector

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
LOCALIZATION_DIR = DATA_DIR / "localization"
IMAGE_INPUT_DIR = LOCALIZATION_DIR / "assets"
IMAGE_OUTPUT_DIR = LOCALIZATION_DIR / "outputs"
FACE_PLOT_OUTPUT_DIR = LOCALIZATION_DIR / "face-plot"

TOTAL_IMGS = 20

# mouth_open, iou are available for running eblations
def determine_eating(
    generate_bounding_boxes: Callable[[Path], list[BoundingBox]], 
    image_path: Path, 
    bbox_output_path: Path = None,
    face_plot_output_path: Path = None,
    mouth_open: bool = True,
    iou: bool = True
):
    pose_detector = PoseDetector()

    mouth_open = determine_mouth_open(pose_detector, image_path, face_plot_output_path)
    iou_condition, status_msg = determine_iou(generate_bounding_boxes, image_path, bbox_output_path)

    if mouth_open and iou
        return mouth_open and iou_condition, status_msg
    elif mouth_open:
        return mouth_open
    elif iou:
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

    for img_num in range(1, TOTAL_IMGS):
        console.print(f"[orange]processing image #{img_num}...[/orange]")
        image_path = IMAGE_INPUT_DIR / f"test{img_num}.jpg"
        output_path = IMAGE_OUTPUT_DIR / f"test{img_num}.jpg"

        # load in ground truth here
        gt = None

        combined, _ = determine_eating(generate_bounding_boxes, image_path, output_path)
        mouth_only, _ = determine_eating(generate_bounding_boxes, image_path, output_path, mouth_open=True, iou=False)
        iou_only, _ = determine_eating(generate_bounding_boxes, image_path, output_path, mouth_open=False, iou=True)

        if gt == combined:
            combined_cnt += 1
        if gt == mouth_only:
            mouth_open_cnt += 1
        if gt == iou_only:
            iou_cnt += 1

        valid_image_cnt += 1

    mouth_only_accuracy = mouth_open_cnt / valid_image_cnt
    iou_only_accuracy = iou_cnt / valid_image_cnt
    combined_accuracy = combined_cnt / valid_image_cnt

    console.print(f"Mouth (only): {mouth_only_accuracy}")
    console.print(f"IOU (only): {iou_only_accuracy}")
    console.print(f"Mouth & IOU: {combined_accuracy}")
