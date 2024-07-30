### Objective
Given an image/video of a person eating, run an open-source LLM on it & ask what food the person is eating (and then branch out to estimating quantity, etc...)

### More questions
- What they are EATING with?
	- Hands? Fork? Spoon? etc...
- Is our prompt output consistent? In both frame 1 and frame 2 of the video? How can we filter it to get a consistent output?
- How much do we care about food weight?

### Tasks
##### 1) Find videos of people eating food
-  [YouTube](https://research.google.com/youtube8m/explore.html) filter by `"eating"`
- [Videos used by past URAs](https://uofwaterloo-my.sharepoint.com/:f:/g/personal/ctnchan_uwaterloo_ca/EtZpFS_b5vZOkPFXZmPWdYkBESOXwahdUvaFr0QIuZOglQ)

### Tried out so far
- [LAVIS](obsidian://open?vault=Obsidian%20Vault&file=2B%2FURA%2FLAVIS)
	- Inference [here](obsidian://open?vault=Obsidian%20Vault&file=2B%2FURA%2FLAVIS)
	- Pros: 
		- Covers what KIND of food the person is eating
	- Cons:
		- Cannot estimate quantity/weight (always defaults to 1 pound) 
		- Cannot list ingredients of the food (the best it can do is say something very vague like "vegetables" for an image of sushi) 
		- Gets confused and describes the actual person instead of the food (edited)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
	- Inference [here](obsidian://open?vault=Obsidian%20Vault&file=2B%2FURA%2FLLaVA)
##### 05-30
- [x] Ask Chris for shared drive
	- [x] Setup shared drive
- [x] Try out [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/predict.py)
- [ ] Add to what Krish has on discord
	- Found prompts that performed very well
		- Description (great)
		- Ingredients (great)
		- Weight (mediocre)
	- Played around with python files to automate running inference on the questions via the command line

##### 05-30
Plan:
- [Automation]([https://stackoverflow.com/questions/34718533/how-to-run-pythons-subprocess-and-leave-it-in-background](https://stackoverflow.com/questions/34718533/how-to-run-pythons-subprocess-and-leave-it-in-background "https://stackoverflow.com/questions/34718533/how-to-run-pythons-subprocess-and-leave-it-in-background")) for LLaVA
	- `subprocess.Popen`
- Document all results & post on discord
- Look at models for weight estimation


![[Pasted image 20240530154216.png]]


##### Improvements - 05-31
- [ ] Prompts
	- [ ] Add prompt for WHAT the person is eating with
		- [ ] Ask Yuhao for `categories` of utensils that the person could potentially be eating with: `[spoon, fork, hands, chopsticks, etc...]`
	- [x] Ask LLaVA to only include key information in its responses
		- [x] Comma separated list
- [x] Pick 10-15 frames of a video (1s apart) and measure consistency (somehow)
- [x] Weight estimation - prompt


##### Tasks (NutritionVerse) CANCELLED
1. How accurate is the model predicting the tasks
	- [ ] Run on NutritionVerse data (quantitative)
2. How consistent/reliable is the prediction
	- [ ] Run multiple epochs and compare them (qualitative)
	- [ ] Feed results into open-source LLM to ask for contextual consistency

##### Questions for Yuhao - RESOLVED
Result: `5.mp4`
```json
avg weight: 1022.0 grams
weights: 
[1500, 1500, 500, 250, 500, 1000, 1000, 1500, 500, 500, 500, 1000, 500, 500, 300, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 500, 1500, 1500]
```
- Super inconsistent
- Any preferred heuristic/method you'd like us to get the "average weight" consistently?

##### 06-06
1. If you are to give me one prediction instead of predictions from multiple frames, how do you design your algorithm?
	- [x] Ingredients: concatenate all `ingredients` from all frames
	- [x] Weight: test for avg weight today

##### More tasks
- Can we figure out what was EATEN?
- Can we accurately summarize an entire video based on the descriptions given in a frame?
	- Can we use our work as a "preprocessing technique" to train models in the future
	- [x] Given 2 frames from the same video, can we ask the model to pinpoint WHAT has changed between the two frames?
- How accurate is the food that is actually being eaten?
- **ADD: ingredient description + whole food description + utensil description**
	- What are the ingredients?
	- What is the actual food name?
	- [x] **Regenerate:** `data.json` with utensil descriptions
- Focus on Quantitative results for the next couple of weeks
	- [x] **Tailor** prompts and measure how "accurate" each result is
		- [x] Used Algorithm for this
	- Calculate precision in terms of ingredient generation - how many of our model's prediction is accurate?
	- [x] Use more videos
	- [ ] Label Video frames in `data.json`
		- We'd have to manually label/look at the videos to do so
		- Ingredients, Utensils
			- Ground truth list
			- False Positive
			- False Negative

##### Adding videos to ``custom-videos/``
- If downloading a YouTube short, don't worry about trimming the video b/c the duration is pretty limited already
- However, if downloading a YouTube video, choose a small interval of time from the video and use ``--trim`` to trim the video down (this is to limit the number of frames created)

### As of 6/10
- Krish: Llama3
	- Need to improve hallucinations
- Sid: gpt4all
	- Need to make it yap less
	- Frame-by-frame comparison (compare 1 with 2, compare 2 with 3, etc...)

### 6/11
- [x] Tailoring prompts to generate utensil descriptions
- [x] Confirmed with Yuhao on future tasks under More Tasks
- Agreed on using GPT4All (with Llama3-instruct)

```c++
LLM: ["omlette", "egg", "cucumber"]

US: ["carrot", "egg || omlette"]

Different: {LLM: ["cucumber"], US: ["carrot"]}
```
Algorithm to match responses from LLM to OUR responses

### TODO: 6/17
- [ ] Label frame-by-frame in `config-custom-videos/`
- [x] Use spacey for `gpt4all` responses
- [x] Devise parsing algorithm to compare specs of LLM responses vs our "desired" responses

### 6/19

```
LLM: [] // everything LLM identified but is NOT true
GT: [] // everything that is TRUE, but LLM did not identify
```

- Penalize a big difference in length of the lists
- Word-to-word, if they're very far apart, penalize

Algorithm:
```
LLM_cos: []
GT_cos: []
```
1. We iterate through all items in `LLM`. For each i in `LLM`, compare it to every j in `GT`. Find the minimum `cosine` diff, add to `LLM_cos`.
2. We iterate through all items in `GT`. For each j in `GT`, compare it with every i in `LLM`. Find the minimum `cosine` diff, add to `GT_cos`.
3. Score = sum of all elements in `LLM_cos` and `GT_cos`

### TODO - 06/21
How exhaustive is our list going to be? How specific should our prompts be?
- [ ] Ask Yuhao for `categories` of utensils that the person could potentially be eating with: `[spoon, fork, hands, chopsticks, etc...]`
- [ ] We have `0.mp4` and `2.mp4`

### 07/08
Notes from Yuhao:
- [ ] Given a frame, can we accurately classify WHAT is BEING eaten?
	- Hand in mouth?
	- Fork/Spoon in mouth?
	- IDEA (first step): We're thinking of using `DWpose` to "measure" whether someone's mouth is open in a frame/image. 
		- we are using `DWpose` to generate image with facial landmarks
		- we now have to find a way to use these facial landmarks to figure out whether mouth is open or not
- [ ] Give a score based on the labels
	- our score thing is kinda uninterpretable rn
	- maybe we can try to use some more standard/already implemented metric

TODO:
- [ ] Label frame-by-frame in `config-custom-videos/`
- [ ] Using `DWposeDetector`:
	- [ ] First, start by classifying every image whether mouth is open, or not.
		- [ ] [Reference]([https://github.com/mauckc/mouth-open](https://github.com/mauckc/mouth-open "https://github.com/mauckc/mouth-open")) - It expects to see the whole face
	- [ ] Then, brainstorm other ideas TODO

### Speak to Yuhao - 07/10
- Show results in `config-custom-videos/`
	- Any suggestions for a different metric? We have `0.mp4` and `2.mp4`
- Show that we setup `DWpose` so that we can generate images for any given input image
	- Any suggestions on how to determine if mouth is open from the generated images
- Qualtitative: We used `GPT4All` to find the DIFFERENCE between items/ingredients/utensils between 2 consecutive frames

---

### TODO (as of 07/10):
1. Using `DWposeDetector`:
	- First, start by classifying every image whether mouth is open, or not ✅
	- Classify whether fingers are pointing towards mouth
	- Fingers pointing towards mouth && mouth open => person is eating
	- SUBTASKS + ideas:
		- Plot distances between lips across frames in `data.json`
		- Use chin as another landmark
2. In `analysis.py`: FINISHED ✅
	- Output a list of True +ve, False +ve, False -ve lists
3. Categorize utensil outputs from LLaVA  FINISHED ✅ (not perfect, false positives detected - spoon, fork)
	- Find a way to get a list of utensils that we can feed to the model in a prompt
	- Give LLM `categories` of utensils that the person could potentially be eating with: `[spoon, fork, hands, chopsticks, etc...]`
4. `0.mp4` and `2.mp4` look great according to Yuhao, so:
	- label frame-by-frame in ALL `config-custom-videos/` 
	- Why? To measure whether our metric + prompts are "good enough"
5. GPT4All: 🗙 CANCELLED (for now - we plan to use a different approach)
	- Re-engineer prompts to turn outputs into lists (so that we can preprocess)
	- Process LLM outputs accordingly
	- Run our algorithm from `analysis.py` (or make a new one to quantitatively classify difference in frames)
		- We need a ground truth
		- We need a "processable" LLM output

---
#### List
```
utensils = [ 
	"Spoon",
    "Fork",
    "Knife",
    "Chopsticks",
    "Hands",
    "Spork",
    "Ladle",
    "Tongs",
    "Spatula",
    "Straw",
    "Bowl",
    "Cup",
]
```
---

### Potential use case of our project
- When you give a toddler some food, the toddler will not necessarily EAT all the food
	- They could throw food away, not eat it, etc...
- In a preschool, parents could get a summary of "Your toddler ate ___ and ___ today, but threw away "
- Nursery walkable from here, can go there to test


## 7/17
- started working on issue #1 (DWPoseDetector)
- Sid idea: can we extract info about face from their code
	- yes, we can!
	- in `__init__.py`, we have this: `faces = candidate[:,24:92]`
	- to validate and make sure that this line actually corresponds to plots of the face, we created plot_face.py to plot a face array that I printed out
	- we were able to validate that the array corresponds to a plot of a face
	- now, the goal is to figure out what landmarks in the face array actually correspond to the mouth


## 7/19 - Updated TODO (DWPoseDetector)
1. ✅ Given one singular frame, classify whether mouth is open, or not
2. 🗙 ON HOLD 🗙 Classify whether fingers are pointing towards mouth
	- Fingers pointing towards mouth && mouth open => person is eating
3. How do we integrate this into multiple frames across the same video?
	- Run our "mouth open" algorithm across several frames in the video
	- Plot the distribution
	- Use chin as another landmark

4. 🗙 ON HOLD 🗙: `0.mp4` and `2.mp4` look great according to Yuhao, so:
	- label frame-by-frame in ALL `config-custom-videos/`
	- Why? To measure whether our metric + prompts are "good enough"

## 7/22 - Updated TODO (Semantic Information)
When someone is eating, the food instance disappears inside the mouth
1. Explore [Semantic Sam](https://github.com/UX-Decoder/Semantic-SAM) and [Grounding Dino](https://github.com/IDEA-Research/GroundingDINO)

## 7/24 - Design Specification
Chris's detailed breakdown on how we should approach whether or not
![image](https://cdn.discordapp.com/attachments/1231042814893359135/1265048864663142540/IMG_5958.jpg?ex=66a21250&is=66a0c0d0&hm=d7106084778eb0d525391aed62f4ca8c9594263b1c2040795bffddac3a994c7f&)

splitting this into smaller bite-sized chunks: (can convert these into tickets)
- setup GroundingDINO
	- provide "food mouth" as input to GroundingDINO
	- find the closest "food" bounding box to the single "mouth" bounding box (use center of boxes to measure distance)
- using facial landmarks if lips are open or not (make a lip separation distance param)
- eating? = closest food bounding box intersects with mouth bounding box with certain minimum intersection area && mouth is open

where does height threshold come in?

### work in progress
getting GroundingDINO working
	- using source code: https://github.com/IDEA-Research/GroundingDINO
	- using HuggingFace:
		- https://github.com/IDEA-Research/GroundingDINO/issues/321
		- https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino 
		- https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Grounding%20DINO
		- https://huggingface.co/models?other=grounding-dino

progress:
- the [installation for setting up inference in the source code](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) in GroundingDINO's README is a bit involved, so decided to look for better options
- found out in [this issue](https://github.com/IDEA-Research/GroundingDINO/issues/321) that GroundingDINO is available in transformers
- found GroundingDINO-base and GroundingDINO-tiny in Huggingface docs
	- atm we are proceeding with the base model, but in the future, if we have hardware limitations, we could switch to GDINO-tiny
- after experimenting (we plotted both) with both normal transformers and the more convenient pipeline api, the normal method seems to be more reliable and accurate; for future reference, here is what the pipeline api looks like: (this is from [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Grounding%20DINO))
```python
from transformers import pipeline
pipe = pipeline(task="zero-shot-object-detection", model="IDEA-Research/grounding-dino-tiny")
results = pipe('http://images.cocodataset.org/val2017/000000039769.jpg', candidate_labels=[preprocess_caption(text)], threshold=0.3)
print(results)
```
- found a newer version of LLaVA-1.6 (LLaVA-Next) and LLaVA-Next-Video, and am now experimenting with using transformers
- 