import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import pprint
from pathlib import Path
from typing import Literal, get_args
import cv2
from PIL import Image
import numpy as np

models = Literal["IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-base"]
SUPPORTED_MODELS = get_args(models)

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


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

    # TODO: generalize the code to use pipeline? (worth?)
    def generate_bounding_boxes(labels: list, image_path: Path):
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

        return results[0]
    
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
    print(f"image: {type(image)}")

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    cv2.rectangle(image, (x, y - text_h), (x + text_w, y), text_color_bg, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, font_thickness)

    return text_size


def draw_bounding_boxes(
    image_path: Path, 
    bounding_boxes: dict, 
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

    scores = bounding_boxes['scores'].cpu().numpy()
    labels = bounding_boxes['labels']
    boxes = bounding_boxes['boxes'].cpu().numpy()

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


if __name__ == "__main__":
    model_name = "IDEA-Research/grounding-dino-base"
    generate_bounding_boxes = make_get_bounding_boxes(model_name)

    root = Path(__file__).parent
    image_path = Path(root / "cat.jpg")
    output_path = Path(root / "dino.jpg")

    # text = "a cat. a remote control."
    labels = ["a cat", "a remote control"]
    bounding_boxes = generate_bounding_boxes(labels, image_path)
    draw_bounding_boxes(image_path, bounding_boxes, output_path)