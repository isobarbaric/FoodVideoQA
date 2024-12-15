# VLM constants
FRAME_STEP_SIZE = 20
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

# Pose Estimation constants
IOU_THRESHOLD = 0.25
LIP_SEPARATION_THRESHOLD = 8

FOOD_ITEM_IDX = 0 # points to the 0th index in VLM_PROMPTS
VLM_PROMPTS = [
    "Identify only the food items visible in the image. Provide a comma-separated list of food items with no additional descriptions or details. Do not repeat any items in your response.",
    f"Provide a list of cutlery/utensils that the person in the image is eating with, from this list: {UTENSILS}. Only provide a comma-separated list of items with no additional descriptions for each item in your response.",
    "Provide a detailed list of the ingredients of the food in the image. Only include a comma-separated list of items with no additional descriptions for each item in your response.",
    "Provide an approximate estimate the weight of the food in the image IN GRAMS. Format: '? grams'. If you cannot provide an estimate, please write '0 grams'. Provide NOTHING ELSE in your response.",
    "Provide nutritional value (calories, protein, fat, carbohydrates) about the food you see in the image in bullet point format with JUST this information and nothing else: \n - Calories = ? \n - Fats = ?% \n - Protein = ?% \n - Carbohydrates = ?% ",
]
