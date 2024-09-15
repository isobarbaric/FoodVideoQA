import json
import time
from pathlib import Path
from .video_utils import extract_frames
import pprint
import json
from llm.generation.models import make_get_response
from tqdm import tqdm
from typing import Callable
from utils.constants import FRAME_STEP_SIZE, UTENSILS

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LLM_DATA_DIR = DATA_DIR / "llm"
LLM_VIDEO_DIR = LLM_DATA_DIR / "videos"
LLM_FRAME_DIR = LLM_DATA_DIR / "frames"


def _describe_frame(get_response: Callable[[str, Path], str],
                    frame_number: int, 
                    image_file: Path, 
                    questions: list[str]):
    """
    Generate descriptions for a specific frame by asking a series of questions about the image.

    Args:
        get_response (Callable[[str, Path], str]): Function to obtain responses from a model given a prompt and an image.
        frame_number (int): The number of the frame being described.
        image_file (Path): Path to the image file of the frame.
        questions (list[str]): List of questions to ask about the frame.

    Returns:
        dict: A dictionary containing the frame number and answers to each question.
    
    Raises:
        ValueError: If the image file does not exist.
    """
    if not image_file.exists():
        raise ValueError(f"No image exists at {image_file.absolute()}")

    answers = {}

    answers['frame_number'] = frame_number
    answers['questions'] = []

    for question in questions:
        actual_response = get_response(question, image_file)
        answers['questions'].append({'prompt': question, 'answer': actual_response})
    
    return answers


def _describe_video(model_name: str,
                    questions: list[str], 
                    video_path: Path, 
                    frame_dir: Path, 
                    frame_step_size: int = 10):
    """
    Process a video by extracting frames, describing each frame, and compiling the results.

    Args:
        model_name (str): The name of the model to use for generating descriptions.
        questions (list[str]): List of questions to ask about each frame.
        video_path (Path): Path to the video file.
        frame_dir (Path): Directory where extracted frames will be saved.
        frame_step_size (int, optional): Interval for extracting frames from the video. Defaults to 10.

    Returns:
        dict: A dictionary containing the video name and descriptions of each frame.
    
    Raises:
        ValueError: If the video file does not exist.
    """
    if not video_path.exists():
        video_path.mkdir(parents=True)
    
    extract_frames(video_path, frame_dir, frame_step_size)
    images = []

    get_response = make_get_response(model_name)
    for frame in sorted(frame_dir.iterdir()):
        print(f"- processing {frame.name}...")
        frame_num = int(frame.name[frame.name.find('frame')+5:frame.name.find('.')])
        current_frame = _describe_frame(get_response, frame_num, frame, questions)
        images.append(current_frame)

    answer = {
        'video_name': video_path.name,
        'frames': images
    }

    return answer
        

def process_videos(model_name: str,
                   video_dir: Path,
                   frame_dir: Path,
                   questions: list[str],
                   output_file: Path,
                   frame_step_size: int = FRAME_STEP_SIZE):
    """
    Process all videos in a directory by extracting frames, generating descriptions, and saving the results to a file.

    Args:
        model_name (str): The name of the model to use for generating descriptions.
        video_dir (Path): Directory containing video files to process.
        frame_dir (Path): Directory where extracted frames will be saved.
        questions (list[str]): List of questions to ask about each frame.
        output_file (Path): Path to the file where results will be saved.
        frame_step_size (int, optional): Interval for extracting frames from the video. Defaults to FRAME_STEP_SIZE.

    Returns:
        list[dict]: A list of dictionaries containing descriptions of each video.
    
    Raises:
        ValueError: If the video directory does not exist.
    """
    if not video_dir.exists():
        raise ValueError(f"Provided file path {video_dir} does not exist")

    answers = []
    for video in tqdm(sorted(video_dir.iterdir())):
        print(f"\nprocessing {video.name}..")
        if video.suffix in ['.mp4']:
            # generalize this to
            frame = frame_dir / video.name
            answers.append(_describe_video(model_name, questions, video, frame, frame_step_size))

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
        f"Provide a list of cutlery/utensils that the person in the image is eating with, from this list: {UTENSILS}.",
        f"Analyze the provided image and provide a list of which utensils are in the image from this list: {UTENSILS}." ,
        "Provide a detailed list of the ingredients of the food in the image. Only include a comma-separated list of items with no additional descriptions for each item in your response.",
        "Provide an approximate estimate the weight of the food in the image in grams. It is completely okay if your estimate is off, all I care about is getting an estimate. Only provide a number and the unit in your response."
    ]

    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    process_videos(model_name, video_dir, frame_dir, questions, output_file, frame_step_size = 20)

    """
    temporary testing code goes below this line
    """

    # sample_frame = Path("extracted-frames/4.mp4/frame15.jpg")

    # questions = [
    #    f"Provide a list of cutlery/utensils that the person in the image is eating with, from this list: {utensils}.",
    # ]

    # response = describe_frame(20, sample_frame, questions)
    # pprint.pprint(response)

    """
    temporary testing code ends above this line
    """

    end = time.time()    
    print(f"\n{round(end - start, 2)} seconds elapsed...")