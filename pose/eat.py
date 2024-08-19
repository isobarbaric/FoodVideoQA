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


def determine_eating(
    generate_bounding_boxes: Callable[[Path], list[BoundingBox]], 
    pose_detector: PoseDetector, 
    image_path: Path, 
    output_path: Path = None
):
    mouth_open = determine_mouth_open(pose_detector, image_path, output_path)
    iou_condition, status_msg = determine_iou(generate_bounding_boxes, image_path, output_path)
    return mouth_open and iou_condition, status_msg


if __name__ == "__main__":
    console = Console()

    model_name = "IDEA-Research/grounding-dino-base"
    generate_bounding_boxes = make_get_bounding_boxes(model_name)
    pose_detector = PoseDetector()

    for img_num in range(1, 20):
        console.print(f"[orange]processing image #{img_num}...[/orange]")
        image_path = IMAGE_INPUT_DIR / f"test{img_num}.jpg"
        output_path = IMAGE_OUTPUT_DIR / f"test{img_num}.jpg"

        eating, msg = determine_eating(generate_bounding_boxes, pose_detector, image_path, output_path)
        if eating:
            console.print(f"[green]{msg}[/green]")
        else:
            console.print(f"[red]{msg}[/red]")
        
        console.print()
        
