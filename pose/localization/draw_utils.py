import numpy as np
import cv2
from pathlib import Path
from pose.localization.bbox import BoundingBox


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


def draw_bounding_boxes(
    image_path: Path, 
    bounding_boxes: list[BoundingBox], 
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

    for bbox in bounding_boxes:
        img_label = f'{bbox.label}: {bbox.score:0.2f}'
        x, y = round(bbox.xmin), round(bbox.ymin)

        # w = change in x
        w = round(bbox.xmax) - round(bbox.xmin)
        # h = change in y
        h = round(bbox.ymax) - round(bbox.ymin)

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