import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import pprint
import matplotlib.pyplot as plt
import numpy as np

model_id = "IDEA-Research/grounding-dino-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
# Check for cats and remote controls
text = "a cat. a remote control."

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

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100

    pprint.pprint(scores)
    pprint.pprint(labels)
    pprint.pprint(boxes)

    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        img_label = f'{label}: {score:0.2f}'
        ax.text(xmin, ymin, img_label, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig('plot-main.jpg')

results = results[0]

pprint.pprint(results)
plot_results(image, results['scores'].cpu().numpy(), results['labels'], results['boxes'].cpu().numpy())

# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import cv2

# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
# IMAGE_PATH = "chai.jpg"
# TEXT_PROMPT = "two dogs with a stick"
# BOX_TRESHOLD = 0.35
# TEXT_TRESHOLD = 0.25

# image_source, image = load_image(IMAGE_PATH)

# boxes, logits, phrases = predict(
#     model=model,
#     image=image,
#     caption=TEXT_PROMPT,
#     box_threshold=BOX_TRESHOLD,
#     text_threshold=TEXT_TRESHOLD
# )

# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# cv2.imwrite("annotated_image.jpg", annotated_frame)