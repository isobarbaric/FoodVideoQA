import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import pprint
from pathlib import Path
from typing import Literal, Callable, get_args
from PIL import Image
from rich.console import Console
from pose.localization.bbox import BoundingBox, Labels
from pose.localization.bbox_utils import get_food_bboxes, get_closest_food_bbox, get_mouth_bbox, bbox_intersection, get_furthest_food_bbox
from pose.localization.draw_utils import draw_bounding_boxes
from utils.constants import IOU_THRESHOLD
import cv2
import numpy as np

# TODO: make video and frame folder under data/ directory
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LOCALIZATION_DIR = DATA_DIR / "localization"
IMAGE_INPUT_DIR = LOCALIZATION_DIR / "assets"
IMAGE_OUTPUT_DIR = LOCALIZATION_DIR / "outputs"

models = Literal["IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-base"]
SUPPORTED_MODELS = get_args(models)


def get_model(model_name: str):
    """
    Load a zero-shot object (GroundingDino) detection model and processor.

    Args:
        model_name (str): Name of the model to load.
    Raises:
        ValueError: If the model is not supported.
    Returns:
        tuple: (processor, model, device) where:
            - processor (AutoProcessor): The model processor.
            - model (AutoModelForZeroShotObjectDetection): The pre-trained model.
            - device (torch.device): Device where the model is loaded (CUDA or CPU).
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"{model_name} model not supported; supported models are {SUPPORTED_MODELS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)

    return processor, model, device


def preprocess_labels(labels: list) -> str:
    """
    Format labels into a string suitable for GroundingDINO.

    Args:
        labels (list): A list of labels (strings) to be formatted.
    Returns:
        str: A single formatted string where labels are joined by '. ' and end with a '.'.
    """
    combined_labels = ". ".join(labels) + '.'
    return combined_labels


def process_bounding_boxes(bounding_boxes_data: dict) -> list[BoundingBox]:
    scores = bounding_boxes_data['scores'].cpu().numpy()
    labels = bounding_boxes_data['labels']
    boxes = bounding_boxes_data['boxes'].cpu().numpy()

    data = []
    for score, label, bbox in zip(scores, labels, boxes):
        data.append(BoundingBox(bbox, label, score))

    return data


def make_get_bounding_boxes(model_name: str):
    f"""
    Create a function for generating bounding boxes using the specified model.

    Args:
        model_name (str): Name of the model to use. Supported models include {SUPPORTED_MODELS}
    Returns:
        function: A function that takes labels and an image path, and returns bounding boxes for the objects.
    """
    processor, model, device = get_model(model_name)

    # TODO: make this modular for some model
    def generate_bounding_boxes(image_path: Path, labels: list = [Labels.MOUTH, Labels.FOOD]) -> BoundingBox:
        """
        Generate bounding boxes for the given labels in the image.

        Args:
            labels (list): List of labels to detect.
            image_path (Path): Path to the image.
        Returns:
            dict: Bounding box results.
        """
        text = preprocess_labels(labels)
        image = Image.open(image_path)
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        bboxes_data = results[0]
        # pprint.pprint(bboxes_data)
        return process_bounding_boxes(bboxes_data)
    
    return generate_bounding_boxes


# TODO: setup logger for errors
def determine_iou(
    generate_bounding_boxes: Callable[[Path], list[BoundingBox]], 
    image_path: Path, 
    output_path: Path = None
):
    if not image_path.exists():
        raise ValueError(f"No image found at image path {image_path}")

    bounding_boxes = generate_bounding_boxes(image_path)
    image = cv2.imread(image_path)

    if output_path is not None:
        if not output_path.exists():
            output_path.touch()
        image = draw_bounding_boxes(image, bounding_boxes)
        cv2.imwrite(str(output_path), image)

    try:
        mouth_bbox = get_mouth_bbox(bounding_boxes)
    except Exception as e:
        return False, str(Exception(e))

    try:
        food_bboxes = get_food_bboxes(bounding_boxes)
    except Exception as e:
        return False, str(Exception(e))
    
    try:
        closest_food_bbox = get_closest_food_bbox(mouth_bbox, food_bboxes)
    except Exception as e:
        return False, str(Exception(e))
    
    try:
        furthest_food_bbox = get_furthest_food_bbox(mouth_bbox, food_bboxes)
    except Exception as e:
        return False, str(Exception(e))

    if output_path is not None:
        image = draw_bounding_boxes(image, [closest_food_bbox, mouth_bbox])
        cv2.imwrite(str(output_path), image)

    iou = bbox_intersection(mouth_bbox, closest_food_bbox)

    condition = iou >= IOU_THRESHOLD
    if condition:
        status_msg = 'iou threshold met'
    else:
        status_msg = 'iou threshold not met'

    return condition, status_msg


if __name__ == "__main__":
    console = Console()

    model_name = "IDEA-Research/grounding-dino-base"
    generate_bounding_boxes = make_get_bounding_boxes(model_name)

    for img_num in range(1, 20):
        console.print(f"[orange]processing image #{img_num}...[/orange]")
        image_path = IMAGE_INPUT_DIR / f"test{img_num}.jpg"
        output_path = IMAGE_OUTPUT_DIR / f"test{img_num}.jpg"

        condition, status_msg = determine_iou(generate_bounding_boxes, image_path, output_path)
        if condition:
            console.print(f"[green]{status_msg}[/green]")
        else:
            console.print(f"[red]{status_msg}[/red]")
        
        console.print()
