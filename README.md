# FoodVideoQA: A Novel Framework for Dietary Monitoring

[![arXiv](https://img.shields.io/badge/CVIS_Publication-Coming_Soon-1eaaaf?logo=livejournal&logoColor=1eaaaf)](https://openjournals.uwaterloo.ca/index.php)
[![arXiv](https://img.shields.io/badge/Arxiv-Coming_Soon-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs)
[![License](https://img.shields.io/badge/Code%20License-Creative_Commons_NC-gree)](https://github.com/isobarbaric/FoodVideoQA/blob/main/LICENSE)

## 🔥 Highlights

- **Cost-Effective**: Uses pre-trained vision-language models without requiring fine-tuning, expensive GPUs, or specialized datasets.
- **Context-Aware Analysis**: Detects foods, utensils, and eating actions frame-by-frame for accurate tracking throughout video input.
- **Domain Adaptable/Scalable**: Provides labeled dietary insights applicable to healthcare, childcare, and assisted living environments without additional equipment.

## 🚀 Functionality  

![Workflow Image](https://github.com/isobarbaric/FoodVideoQA/blob/main/assets/VLM_Image.png)

### 🧩 VLM-Driven Insights
Extracts nutritional information, ingredients, and utensils from video frames using Vision-Language Models. Groups frames into intervals based on consistent food item presence. The [code](https://github.com/isobarbaric/FoodVideoQA/blob/main/vlm/generation/models.py) can be modified to accomodate any of the following HuggingFace VLMs:

- `liuhaotian/llava-v1.5-7b`
- `llava-hf/llava-1.5-7b-hf`
- `llava-hf/llava-v1.6-mistral-7b-hf`
- `Salesforce/blip2-opt-2.7b`

### 🤖 Pose Estimation
Detects eating behavior by checking if the mouth is open and if food is near the mouth using bounding boxes and pose landmarks. We use [DWPose](https://github.com/IDEA-Research/DWPose) to detect mouth landmarks, and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) to localize food items.

#### Example frame of a person eating:
<img src="https://github.com/isobarbaric/FoodVideoQA/blob/main/assets/eating.png" width="400">

#### Example frame of a person NOT eating:
<img src="https://github.com/isobarbaric/FoodVideoQA/blob/main/assets/not-eating.png" width="400">

#### Example face plot using DWPose:
<img src="https://github.com/isobarbaric/FoodVideoQA/blob/main/assets/face-plot.png" height="400">


## 🔧 Hyperparameters  

| **Hyperparameter**          | **Symbol** | **Value**        |  
|------------------------------|------------|------------------|  
| Frame Step Size              | $\tau$     | 20 frames        |  
| Frame Tolerance Threshold    | $\epsilon$ | 15 frames        |  
| Lip Separation Threshold     | $\beta$    | 8.0              |  
| IoU Threshold                | $\delta$   | 0.15             |  

**View and modify hyperparameters** [here](https://github.com/isobarbaric/FoodVideoQA/blob/main/hyperparameters.py).

## 🙏 Acknowledgements
- This work was supported by the [National Research Council Canada (NRC)](https://nrc.canada.ca/en) through the
[Aging in Place (AiP) Challenge Program](https://nrc.canada.ca/en/research-development/research-collaboration/programs/aging-place-challenge-program). Project number **AiP-006**.

- The authors thank the [Vision and Image Processing Lab (VIP Lab)](https://uwaterloo.ca/vision-image-processing-lab/) at the University of Waterloo for facilitating this project.


