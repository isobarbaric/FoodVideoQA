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

def get_response(image_file, questions, verbose=False):
    model_path = "liuhaotian/llava-v1.5-7b"
    command = [
        "python3", "-m", "llava.serve.cli",
        "--model-path", model_path,
        "--image-file", image_file
    ]
    command.append("--load-4bit")
    # command.append("--load-8bit")
    
    answers = {}

    answers['image_file'] = image_file

    for question in questions:
        answers[question] = ''

        # if verbose, then have subprocess.run instead of subprocess.Popen
        result = subprocess.Popen(command, input=question, stdout=subprocess.PIPE, text=True)
        
        result = result.stdout

        # Split the string based on "ASSISTANT:"
        split_result = result.split("ASSISTANT:")

        # Get the second part of the split, which contains the actual response
        actual_response = split_result[1].strip()

        # Check if the response ends with "USER:"
        if actual_response.find("USER:") != -1:
            user_index = actual_response.find("USER:")
            actual_response = actual_response[:user_index]

        actual_response = actual_response.strip()
        print(f"'{actual_response}'")

        answers[question] = actual_response
    
    return answers

image_file = Path("custom-images") / "bread.png"
# get_response(image_file, question="Provide a detailed description of the food you see in the image.")

questions = [
    "1) Provide a detailed description of the food you see in the image.",
    "2) Provide a detailed list of the ingredients of the food in the image.",
    "3) Provide an approximate estimate the weight of the food in the image in grams. It is completely okay if your estimate is off, all I care about is getting an estimate."
]

get_response(image_file, questions)

# for question in questions:
#     get_response(image_file, question)
