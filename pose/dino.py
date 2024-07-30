import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import pprint
from pathlib import Path
from typing import Literal, get_args
import cv2
from PIL import Image
import numpy as np
from dataclasses import dataclass

ROOT = Path(__file__).parent.parent
IMAGE_DIR = ROOT / "data" / "llm" / "misc-images"
INPUT_DIR = IMAGE_DIR / "input"
OUTPUT_DIR = IMAGE_DIR / "output"

models = Literal["IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-base"]
SUPPORTED_MODELS = get_args(models)


def get_model(model_name: str):
    """
    Load a zero-shot object detection model and processor.

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


def make_get_bounding_boxes(model_name: str):
    """
    Create a function for generating bounding boxes using the specified model.

    Args:
        model_name (str): Name of the model to use.
    Returns:
        function: A function that takes labels and an image path, and returns bounding boxes for the objects.
    """
    processor, model, device = get_model(model_name)

    def generate_bounding_boxes(labels: list, image_path: Path) -> BoundingBox:
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


def draw_text(
    image: np.ndarray, 
    text: str,
    pos: tuple[int, int],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.35,
    font_thickness: int = 1,
    text_color: tuple[int, int, int] = (0, 0, 0),
    text_color_bg: tuple[int, int, int] = (31, 132, 187)
):
    """
    Draw text on an image with a background rectangle.

    Args:
        image (ndarray): The image on which to draw the text.
        text (str): The text to be drawn.
        pos (tuple): The bottom-left corner position (x, y) of the text.
        font (int, optional): Font type. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float, optional): Font scale factor. Defaults to 0.35.
        font_thickness (int, optional): Thickness of the text. Defaults to 1.
        text_color (tuple, optional): Color of the text (B, G, R). Defaults to (0, 0, 0).
        text_color_bg (tuple, optional): Background color of the text (B, G, R). Defaults to (31, 132, 187).
    Returns:
        tuple: Size of the drawn text.
    """
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    cv2.rectangle(image, (x, y - text_h), (x + text_w, y), text_color_bg, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, font_thickness)

    return text_size


# TODO: refactor this to use list[DINOBoundingBox]
def draw_bounding_boxes(
    image_path: Path, 
    bounding_boxes_data: dict, 
    output_path: Path,
    show: bool = False
):
    """
    Draw bounding boxes on an image and save or display the result.

    Args:
        image_path (Path): Path to the input image.
        bounding_boxes (dict): Dictionary containing 'scores', 'labels', and 'boxes' for detected objects.
        output_path (Path): Path to save the output image with bounding boxes.
        show (bool, optional): Whether to display the image with bounding boxes. Defaults to False.
    """
    image = cv2.imread(str(image_path))

    scores = bounding_boxes_data['scores'].cpu().numpy()
    labels = bounding_boxes_data['labels']
    boxes = bounding_boxes_data['boxes'].cpu().numpy()

    for score, label, (xmin, ymin, xmax, ymax) in zip(scores, labels, boxes):
        img_label = f'{label}: {score:0.2f}'
        x, y = round(xmin), round(ymin)

        # w = change in x
        w = round(xmax) - round(xmin)
        # h = change in y
        h = round(ymax) - round(ymin)

        cv2.rectangle(image, (x, y), (x+w, y+h), color=(36, 80, 203), thickness=2)
        draw_text(image, img_label, (x, y))

    if not output_path.exists():
        output_path.touch()
    cv2.imwrite(str(output_path), image)
        
    if show:
        img = cv2.imread(str(output_path))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


@dataclass
class BoundingBox:
    bounding_box: np.ndarray
    label: str
    score: float

    @property
    def xmin(self) -> float:
        return self.bounding_box[0]

    @property
    def ymin(self) -> float:
        return self.bounding_box[1]

    @property
    def xmax(self) -> float:
        return self.bounding_box[2]

    @property
    def ymax(self) -> float:
        return self.bounding_box[3]

    @property
    def center(self) -> tuple[int, int]:
        xmin, ymin, xmax, ymax = self.bounding_box
        x_mid = (xmin + xmax) / 2
        y_mid = (ymin + ymax) / 2
        return x_mid, y_mid


def process_bounding_boxes(bounding_boxes_data: dict) -> list[BoundingBox]:
    scores = bounding_boxes_data['scores'].cpu().numpy()
    labels = bounding_boxes_data['labels']
    boxes = bounding_boxes_data['boxes'].cpu().numpy()

    data = []
    for score, label, bbox in zip(scores, labels, boxes):
        data.append(BoundingBox(bbox, label, score))

    return data


def get_mouth_bbox(bounding_boxes: list[BoundingBox]):
    bboxes = [bbox_obj for bbox_obj in bounding_boxes if bbox_obj.label == 'mouth']
    
    if len(bboxes) == 0:
        raise IndexError(f"No bounding box found associated with label 'mouth'")

    if len(bboxes) > 1:
        raise ValueError(f"More than one bounding box found with label 'mouth'")

    return bboxes[0]


def get_food_bboxes(bounding_boxes: list[BoundingBox]):
    bboxes = [bbox_obj for bbox_obj in bounding_boxes if bbox_obj.label == 'food']
    return bboxes


if __name__ == "__main__":
    model_name = "IDEA-Research/grounding-dino-base"
    generate_bounding_boxes = make_get_bounding_boxes(model_name)

    image_path = INPUT_DIR / "eat-2.jpg"
    output_path = OUTPUT_DIR / "dino-2.jpg"

    labels = ["food", "mouth"]
    bounding_boxes = generate_bounding_boxes(labels, image_path)
    print("all bounding boxes:")
    pprint.pprint(bounding_boxes)

    # draw_bounding_boxes(image_path, bounding_boxes, output_path)

    mouth_bbox = get_mouth_bbox(bounding_boxes)
    print("\nmouth bounding box:")
    pprint.pprint(mouth_bbox)

    food_bboxes = get_food_bboxes(bounding_boxes)
    print("\nfood bounding boxes:")
    pprint.pprint(food_bboxes)