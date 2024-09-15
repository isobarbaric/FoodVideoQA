# NutritionVerse-LLM

>[Video metadata](https://docs.google.com/spreadsheets/d/1WNfDNICa5WdvEl8qDeNTEkO1e3Mvv6gJKwoE2ofs81w/edit?usp=sharing)

## Main Features
1. Food & Utensil Detection
2. Food & Utensil Analysis
3. Food items eaten throughout video
4. Eating Action Detection


## Folder Structure
```bash
NutritionVerse-LLM/
├── README.md
├── data/
│   ├── detection/
│   │   ├── assets/
│   │   ├── face-plot/
│   │   └── inference/
│   ├── llm/
│   │   ├── frames/
│   │   ├── videos/
│   │   ├── data.json
│   │   └── frame_diff.json
│   └── localization/
│       ├── assets/
│       └── outputs/
├── llm/
│   ├── analysis/
│   │   ├── analyze.py
│   │   ├── parser.py
│   │   └── scoring.py
│   ├── frame_diff/
│   │   └── frames.py
│   └── generation/
│       ├── generate.py
│       ├── models.py
│       ├── video_utils.py
│       ├── youtube.py
│       └── eaten.py
├── pose/
│   ├── detection/
│   │   ├── ckpts/
│   │   ├── config/
│   │   ├── drawing_utils.py
│   │   ├── face_plotting.py
│   │   ├── inference_utils.py
│   │   ├── inference.py
│   │   ├── wholebody.py
│   │   ├── requirements.txt
│   │   └── pose_detector.py   
│   └── localization/
│       ├── bbox_utils.py
│       ├── bbox.py
│       ├── dino.py
│       ├── draw_utils.py
│       └── eat.py
└── utils/
    └── constants.py

```
---

### 🔍 Food & Utensil Detection - `llm/generation/`
**Functionality**: 
Identifies and describes the food being consumed, and utensils being used to eat, for each frame in a video.

**Implementation**: 
- Takes input from `data/llm/videos`
- Splits each video into frames, outputs frames to `data/llm/frames/`
- Uses `LLaVA` to generate descriptions of food being consumed, and outputs result to `data/llm/data.json`


In `generate.py`, a list of prompts is provided to `LLaVA`:

```python
    questions = [
        "Provide a detailed description of the food you see in the image.",
        f"Provide a list of cutlery/utensils that the person in the image is eating with, from this list: {UTENSILS}.",
        f"Analyze the provided image and provide a list of which utensils are in the image from this list: {UTENSILS}." ,
        "Provide a detailed list of the ingredients of the food in the image. Only include a comma-separated list of items with no additional descriptions for each item in your response.",
        "Provide an approximate estimate the weight of the food in the image in grams. It is completely okay if your estimate is off, all I care about is getting an estimate. Only provide a number and the unit in your response."
    ]
```
where `UTENSILS` is imported from `utils/constants.py`

A sample frame is stored as follows in `data.json`:
```json
[
    {
        "video_name": "video_1.mp4",
        "frames": [
            ...
            {
                "frame_number": 1020,
                "questions": [
                    {
                        "prompt": "Provide a detailed description of the food you see in the image.",
                        "answer": "The image shows a person holding a piece of fried chicken over a bowl filled with more fried chicken. The chicken appears to be golden brown, indicating it has been fried until crispy. The person is also holding a piece of fried chicken in their other hand. The setting seems to be indoors, possibly a restaurant or a home kitchen, as suggested"
                    },
                    {
                        "prompt": "Provide a list of cutlery/utensils that the person in the image is eating with, from this list: ['spoon', 'fork', 'knife', 'chopstick', 'spork', 'ladle', 'tongs', 'spatula', 'straw', 'bowl', 'cup', 'glass'].",
                        "answer": "The person in the image is eating with a 'spatula'."
                    },
                    {
                        "prompt": "Analyze the provided image and provide a list of which utensils are in the image from this list: ['spoon', 'fork', 'knife', 'chopstick', 'spork', 'ladle', 'tongs', 'spatula', 'straw', 'bowl', 'cup', 'glass'].",
                        "answer": "The image provided shows a person holding a piece of fried chicken. There are no utensils visible in the image. The items present in the image are a piece of fried chicken, a bowl containing more fried chicken, and a person's hand holding the chicken."
                    },
                    {
                        "prompt": "Provide a detailed list of the ingredients of the food in the image. Only include a comma-separated list of items with no additional descriptions for each item in your response.",
                        "answer": "chicken, breading, oil, salt, pepper"
                    },
                    {
                        "prompt": "Provide an approximate estimate the weight of the food in the image in grams. It is completely okay if your estimate is off, all I care about is getting an estimate. Only provide a number and the unit in your response.",
                        "answer": "Based on the image, it is difficult to provide an accurate weight estimate for the food. However, I can give you a rough estimate. The bowls are filled with what appears to be fried chicken, and there are several pieces visible. Assuming an average weight for a piece of fried chicken to be around 100 grams, we can estimate that"
                    }
                ]
            }
            ... (more frames)
        ]
    }
    ... (more videos)
]
```

---
### 🔬 Food & Utensil Analysis - `llm/analysis/`
**Functionality**: 
Evaluates the performance of the `LLaVA` against ground truth data (manually labelled by us) to assess the accuracy of food and utensil detection and identification (from the previous detection step).

The goal here is to make sure `LLaVA` is a feasible modlel to use for huge amounts of video data.

**Implementation**: 
- Takes input from `data/llm/data.json`
- `compare_pred()` compares the LLM's output with ground truth data to identify discrepancies\
- `match_output()` matches LLM outputs to ground truth lists.
- `compute_diff_score()` computes a score based on the semantic similarity between false positive and false negative terms using Word2Vec model distances
- Outputs a diff report: identifies **true positives**, **false positives**, and **false negatives**

#### Usage (in code)
```python
if __name__ == "__main__":
    # test 1 -- Dummy data
    llm_lst = ["omelette", "egg", "cucumber", "fries", "potato", "tomato"]
    gt_lst = ["carrot", "egg || omelette", "tomatoes || tomato", "potato || fries || potato fries"]

    diff = match_outputs(llm_lst, gt_lst)
    score = compute_diff_score(diff)

    print(f"Score: {score}, Diff: {diff}")


    # test 2 -- Actual data
    diff, score = compare_pred(video_name='2.mp4', frame_number=15, prompt_index=INGREDIENTS_PROMPT_INDEX)
    print(f"Score: {score}, Diff: {diff}")
```


---
### Food Items Eaten Throughout Video - `llm/frame_diff/`
**Functionality**: 
- Tracks and identifies food items consumed/introduced on a frame-by-frame basis.
- Summarizes all food items consumed throughout video. 

**Implementation**: 

Files Involved:
llm/frame_diff/frames.py: Handles frame differences to track food consumption over time.
data/llm/frame_diff.json: Stores information about changes between frames.
llm/analysis/: Contains scripts like analyze.py for summarizing and scoring the food items consumed.


- Takes input from `data/llm/data.json`
- Uses `GPT4All`'s `Meta-Llama-3-8B` model to analyze a **previous frame** and a **current frame** 
- Writes output into `data/llm/frame_diff.json`


Prompt used to analyze difference:
```python
frame_diff_prompt =   """
You are analyzing changes between two consecutive frames in a video. Focus ONLY on the presence of NEW food items and the absence of OLD food items between the two descriptions.

Please provide your answer in the following structured distionary format:

{
    "new": [List of food items that appear in the current frame but were not present in the previous frame],
    "absent": [List of food items that were present in the previous frame but are missing in the current frame]
}

Only mention the food items that are NEW or ABSENT. If there are none in either list, leave the list empty. Do NOT include any other information in your response.
"""
```

TODO:
- Need to incorporate a parser (from word2vec) since LLM sometimes outputs missing utensils

#### Usage:
Run the following command from the `NutritionVerse-LLM` directory:
```bash
python3 -m llm.eaten
```
---

