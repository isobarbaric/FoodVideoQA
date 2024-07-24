import json
import time
import subprocess
from pathlib import Path
from .video import extract_frames
import pprint
import json
from .llava import get_response 
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LLM_DATA_DIR = DATA_DIR / "llm"
LLM_VIDEO_DIR = LLM_DATA_DIR / "videos"
LLM_FRAME_DIR = LLM_DATA_DIR / "frames"

utensils = [ 
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

def describe_frame(frame_number: int, 
                   image_file: Path, 
                   questions: list[str]):
    if not image_file.exists():
        raise ValueError(f"No image exists at {image_file.absolute()}")

    answers = {}

    answers['frame_number'] = frame_number
    answers['questions'] = []

    for question in questions:
        actual_response = get_response(question, image_file)
        answers['questions'].append({'prompt': question, 'answer': actual_response})
    
    return answers


def describe_video(questions: list[str], 
                   video_path: Path, 
                   frame_dir: Path, 
                   k: int = 10):
    if not video_path.exists():
        raise ValueError(f"Provided file path {video_path} does not exist")
    
    extract_frames(video_path, frame_dir, k)
    images = []

    for frame in sorted(frame_dir.iterdir()):
        print(f"- processing {frame.name}...")
        frame_num = int(frame.name[frame.name.find('frame')+5:frame.name.find('.')])
        current_frame = describe_frame(frame_num, frame, questions)
        images.append(current_frame)

    answer = {
        'video_name': video_path.name,
        'frames': images
    }

    return answer
        

def process_videos(video_dir: Path,
                   frame_dir: Path,
                   questions: list[str],
                   output_file: Path,
                   k: int = 10):
    if not video_dir.exists():
        raise ValueError(f"Provided file path {video_dir} does not exist")

    answers = []
    for video in tqdm(sorted(video_dir.iterdir())):
        print(f"\nprocessing {video.name}..")
        if video.suffix in ['.mp4']:
            # generalize this to
            frame = frame_dir / video.name
            answers.append(describe_video(questions, video, frame, k))

        with open(output_file, 'w') as f:
            json.dump(answers, f, indent=4) 

    return answers


if __name__ == "__main__":
    start = time.time()    

    output_file = LLM_DATA_DIR / "data.json"
    video_dir = LLM_VIDEO_DIR
    frame_dir = LLM_FRAME_DIR

    questions = [
        "Provide a detailed description of the food you see in the image.",
        f"Provide a list of cutlery/utensils that the person in the image is eating with, from this list: {utensils}.",
        f"Analyze the provided image and provide a list of which utensils are in the image from this list: {utensils}." ,
        "Provide a detailed list of the ingredients of the food in the image. Only include a comma-separated list of items with no additional descriptions for each item in your response.",
        "Provide an approximate estimate the weight of the food in the image in grams. It is completely okay if your estimate is off, all I care about is getting an estimate. Only provide a number and the unit in your response."
    ]

    process_videos(video_dir, frame_dir, questions, output_file, k = 20)

    ###
    # temporary testing code goes below this line    
    ###

    # sample_frame = Path("extracted-frames/4.mp4/frame15.jpg")

    # questions = [
    #    f"Provide a list of cutlery/utensils that the person in the image is eating with, from this list: {utensils}.",
    # ]

    # response = describe_frame(20, sample_frame, questions)
    # pprint.pprint(response)

    ###
    # temporary testing code ends above this line    
    ###

    end = time.time()    
    print(f"\n{round(end - start, 2)} seconds elapsed...")