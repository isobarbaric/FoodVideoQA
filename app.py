from pathlib import Path
# from predict import Predictor

# predictor = Predictor()

# raw_image = Path("custom-images") / "biryani.png"
# response = predictor.predict(
#     image = raw_image,
#     prompt = "What are the individual ingredents in the food?" 
# )

# print(response)

# for text in response:
#     print(text)

import subprocess

def get_response(image_file):
    model_path = "liuhaotian/llava-v1.5-7b"
    command = [
        "python3", "-m", "llava.serve.cli",
        "--model-path", model_path,
        "--image-file", image_file
    ]
    command.append("--load-4bit")
    # command.append("--load-8bit")
    
    subprocess.run(command)

image_file = Path("custom-images") / "bread.png"

get_response(image_file)

# questions = [
#     "What are the individual ingredents in the food?",
#     "What is the approximate weight of the food item in grams?"
# ]