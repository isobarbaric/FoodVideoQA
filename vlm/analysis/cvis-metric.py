import json
from pathlib import Path
import pprint

ROOT_DIR = Path(__file__).parent.parent.parent

DATASET_DIR = ROOT_DIR / "dataset"
IMAGE_INPUT_DIR = DATASET_DIR / "frames"
IMAGE_OUTPUT_DIR = DATASET_DIR / "annotated-frames"
GROUND_TRUTH_JSON = DATASET_DIR / "ground_truth.json"

with open(GROUND_TRUTH_JSON, "r") as f:
    ground_truth = json.load(f)

eating_list = []
rest_list = []

for video_name, frame_data in ground_truth.items():
    for frame_name, truth_val in frame_data.items():
        if truth_val == 0:
            rest_list.append([video_name, frame_name])
        else:
            eating_list.append([video_name, frame_name])

total_imgs = 100
eating_imgs = 0

eating_list = eating_list[:eating_imgs]
rest_list = rest_list[: total_imgs - eating_imgs]
print(len(rest_list), len(eating_list))

assert len(eating_list) == eating_imgs
assert len(rest_list) == total_imgs - eating_imgs

imgs = eating_list + rest_list
imgs.sort()
assert len(imgs) == len(imgs)

with open(DATASET_DIR / "filtered-dataset.json", "w") as f:
    json.dump(imgs, f, indent=4)
