# config containing hyperparameters used in project

# VLM constants
FRAME_STEP_SIZE = 10
UTENSILS = [ 
    "spoon",
    "fork",
    "knife",
    "chopstick",
    "spork",
    "ladle",
    "tongs",
    "spatula",
    "straw",
    "bowl",
    "cup",
    "glass"
]

VLM_PROMPTS = [
    "Provide a detailed description of the food you see in the image.",
    f"Provide a list of cutlery/utensils that the person in the image is eating with, from this list: {UTENSILS}. Only provide a comma-separated list of items with no additional descriptions for each item in your response.",
    "Provide a detailed list of the ingredients of the food in the image. Only include a comma-separated list of items with no additional descriptions for each item in your response.",
    "Provide an approximate estimate the weight of the food in the image IN GRAMS. Format: '? grams'. If you cannot provide an estimate, please write '0 grams'. Provide NOTHING ELSE in your response.",
    """Provide nutritional value (calories, protein, fat, carbohydrates) about the food you see in the image in bullet point format with JUST this information and nothing else: 
    - Calories = ?
    - Fats = ?%
    - Protein = ?%
    - Carbohydrates = ?% 
    """
]

FRAME_DIFFERENCE_PROMPT = """
You are analyzing changes between two consecutive frames in a video. Focus ONLY on the presence of NEW food items and the absence of OLD food items between the two descriptions.

Please provide your answer in the following structured dictionary format. ONLY include edible food items in your response.
{
    "new": [List of food items that appear in the current frame but were not present in the previous frame. INCLUDE ONLY EDIBLE PREPARED FOOD ITEMS.],
    "absent": [List of food items that were present in the previous frame but are missing in the current frame. INCLUDE ONLY EDIBLE PREPARED FOOD ITEMS.]
}

Only mention the food items that are NEW or ABSENT. If there is no FOOD information, leave the respective list empty. . Do NOT include any other information in your response.
"""

# Pose Estimation constants
IOU_THRESHOLD = 0
LIP_SEPARATION_THRESHOLD = 8