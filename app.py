from pathlib import Path
import subprocess
import json
from tqdm import tqdm
import time

# TODO: implement verbose feature
def get_response(image_file: Path, 
                 questions: list[str]):
    if not image_file.exists():
        raise ValueError(f"No image exists at {image_file.absolute()}")

    model_path = "liuhaotian/llava-v1.5-7b"
    command = [
        "python3", "-W ignore", "-m",  
        "llava.serve.cli",
        "--model-path", model_path,
        "--image-file", image_file
    ]
    command.append("--load-4bit")
    # command.append("--load-8bit")
    
    answers = {}

    answers['image_path'] = str(image_file)
    answers['questions'] = []

    for question in questions:
        result = subprocess.run(command, input=question, stdout=subprocess.PIPE, text=True)

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
        # print(f"'{actual_response}'")

        answers['questions'].append({'prompt': question, 'answer': actual_response})
    
    return answers


if __name__ == "__main__":
    images_folder = Path("custom-images")

    questions = [
        "Provide a detailed description of the food you see in the image.",
        "Provide a detailed list of the ingredients of the food in the image.",
        "Provide an approximate estimate the weight of the food in the image in grams. It is completely okay if your estimate is off, all I care about is getting an estimate."
    ]

    start = time.time()    

    answers = []
    for img in tqdm(images_folder.iterdir()):
        print(f"\n\nprocessing {img.name}...\n")
        if img.suffix in ['.png', '.jpg']:
            answers.append(get_response(img, questions))

        # keep updating answers in case of a crash
        with open('data.json', 'w') as f:
            json.dump(answers, f, indent=4)

    end = time.time()    
    print(f"{round(end - start, 2)} seconds elapsed...")

