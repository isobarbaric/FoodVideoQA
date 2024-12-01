# FoodVideoQA: A Novel Framework for Dietary Monitoring

[![arXiv](https://img.shields.io/badge/CVIS_Publication-Coming_Soon-1eaaaf?logo=livejournal&logoColor=1eaaaf)](https://openjournals.uwaterloo.ca/index.php)
[![arXiv](https://img.shields.io/badge/Arxiv-Coming_Soon-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs)
[![License](https://img.shields.io/badge/Code%20License-Creative_Commons_NC-gree)](https://github.com/isobarbaric/FoodVideoQA/blob/main/LICENSE)


## 📢 News & Releases
- **[2024/12/01]** We've made the repository **public**. Feel free to **star** ⭐ this repository for our latest updates.

## 🔥 Highlights

- **Zero Training**: FoodVideoQA builds upon state-of-the-art, pre-trained vision-language models (VLMs) and does not require training or fine-tuning.
- **Cost-Effective**: Eliminates dependency on expensive GPU resources with the use of pre-trained models.
- **No Dataset Required**: Avoids the requirement of hand curated datasets that try to capture all possible food and utensil combinations, and allows for real world applicability.  
- **Domain Adaptable**: Generalizes to various food & utensil scenarios without retraining for new tasks or environments.
- **Non-Invasive**:  In contrast to wearable devices, it does not force users to take deliberate action to achieve the output.
- **Temporal Context**: Processes videos frame by frame to ensure coherence across longer videos while recognizing appearance as well as disappearance of food.
- **Eating Detection**: Combines VLM-based insights with pose estimation to identify mouth openness and proximity to food; distinguishes eating actions from occlusions, thanks to [DWPose](https://github.com/IDEA-Research/DWPose).
- **Detailed Analysis**: Provides descriptions of food items, ingredients, utensils, and nutritional information on a **frame-by-frame** basis.
- **Efficient Workflow**: Eating intervals are compiled into labeled outputs providing an easy way to monitor dietary intervals.
- **Scalable**: Designed for real-world applications in healthcare, nurseries, and assisted living environments. Supports dietary behavior analysis **without specialized equipment**.

## 🚀 Functionality  

![Workflow Image](https://github.com/isobarbaric/FoodVideoQA/blob/main/assets/VLM_Image.png)

### 🧩 VLM-Driven Insights
Extracts nutritional information, ingredients, and utensils from video frames using Vision-Language Models. Groups frames into intervals based on consistent food item presence. We used [LLaVA 1.6](https://huggingface.co/liuhaotian/llava-v1.6-34b). However, the [code](https://github.com/isobarbaric/FoodVideoQA/blob/main/vlm/generation/models.py) can be modified to accomodate any of the following models:
```python
[
"liuhaotian/llava-v1.5-7b",
"llava-hf/llava-1.5-7b-hf",
"llava-hf/llava-v1.6-mistral-7b-hf", 
"Salesforce/blip2-opt-2.7b"
]
```

### 🤖 Pose Estimation
Detects eating behavior by checking if the mouth is open and if food is near the mouth using bounding boxes and pose landmarks. We use [DWPose](https://github.com/IDEA-Research/DWPose) to detect mouth landmarks, and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) to localize food items.

Example frame of a person eating:

![Eating](https://github.com/isobarbaric/FoodVideoQA/blob/main/assets/eating.png)

Example frame of a person NOT eating:

![Not Eating](https://github.com/isobarbaric/FoodVideoQA/blob/main/assets/not-eating.png)


### 🔧 Hyperparameters  

| **Hyperparameter**          | **Symbol** | **Value**        |  
|------------------------------|------------|------------------|  
| Frame Step Size              | $\tau$     | 20 frames        |  
| Frame Tolerance Threshold    | $\epsilon$ | 15 frames        |  
| Lip Separation Threshold     | $\beta$    | 8.0              |  
| IoU Threshold                | $\delta$   | 0.15             |  

**View and modify hyperparameters** [here](https://github.com/isobarbaric/FoodVideoQA/blob/main/hyperparameters.py).

## 📊 Dataset
Manually-labelled dataset to validate semantic accuracy of VLMs: **Coming Soon!**



## 🙏 Acknowledgements
- This work was supported by the [National Research Council Canada (NRC)](https://nrc.canada.ca/en) through the
[Aging in Place (AiP) Challenge Program](https://nrc.canada.ca/en/research-development/research-collaboration/programs/aging-place-challenge-program). Project number **AiP-006**.

- The authors thank the [Vision and Image Processing Lab (VIP Lab)](https://uwaterloo.ca/vision-image-processing-lab/) at the University of Waterloo for facilitating this project.


