# ⭐ LLM to estimate Food Quantities ⭐

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
- [ ] Pick 10-15 frames of a video (1s apart) and measure consistency (somehow)
- [ ] Weight estimation


##### Tasks
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
	- [ ] **Regenerate:** `data.json` with utensil descriptions
- Focus on Quantitative results for the next couple of weeks
	- [ ] **Tailor** prompts and measure how "accurate" each result is
	- Calculate precision in terms of ingredient generation - how many of our model's prediction is accurate?
	- Use more videos
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
- Tailoring prompts to generate utensil descriptions
- Confirmed with Yuhao on future tasks under More Tasks
- Agreed on using GPT4All (with Llama3-instruct)