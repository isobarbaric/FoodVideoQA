import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import pprint
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal, get_args

models = Literal["IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-base"]
SUPPORTED_MODELS = get_args(models)

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def get_model(model_name: str):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"{model_name} model not supported; supported models are {SUPPORTED_MODELS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)

    return processor, model, device


def preprocess_labels(labels: list) -> str:
    combined_labels = ". ".join(labels) + '.'
    return combined_labels


def make_get_bounding_boxes(model_name: str):
    processor, model, device = get_model(model_name)

    # TODO: generalize the code to use pipeline? (worth?)
    def generate_bounding_boxes(labels: list, image: Image):
        text = preprocess_labels(labels)
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


def draw_bounding_boxes(
    image: Image, 
    bounding_boxes: dict, 
    output_path: Path = None
):
    plt.figure(figsize=(16,10))
    plt.imshow(image)
    ax = plt.gca()
    colors = COLORS * 100

    scores = bounding_boxes['scores'].cpu().numpy()
    labels = bounding_boxes['labels']
    boxes = bounding_boxes['boxes'].cpu().numpy()

    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        img_label = f'{label}: {score:0.2f}'
        ax.text(xmin, ymin, img_label, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.tight_layout()

    if output_path is not None:
        if not output_path.exists():
            output_path.touch()
        plt.savefig(output_path)
    else:
        plt.show()


if __name__ == "__main__":
    model_name = "IDEA-Research/grounding-dino-tiny"
    generate_bounding_boxes = make_get_bounding_boxes(model_name)

    # text = "a cat. a remote control."
    labels = ["a cat", "a remote control"]
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    bounding_boxes = generate_bounding_boxes(labels, image)

    root = Path(__file__).parent
    output_path = Path(root / "dino.jpg")

    draw_bounding_boxes(image, bounding_boxes, output_path)