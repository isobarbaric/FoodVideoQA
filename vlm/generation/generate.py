import json
import time
from pathlib import Path
from .video_utils import extract_frames, get_frame_rate
import json
from vlm.generation.models import make_get_response
from tqdm import tqdm
from typing import Callable
import pprint
from hyperparameters import VLM_PROMPTS, FRAME_STEP_SIZE
import cv2

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
        print(f"Q: {question}\nA: {actual_response}")
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

    def _get_frame_number(frame_path: Path):
        return int(frame_path.name[frame_path.name.find('frame')+5:frame_path.name.find('.')])

    def _sort_frames(frame_dir: Path):
        return sorted(frame_dir.iterdir(), key=lambda x: _get_frame_number(x))

    get_response = make_get_response(model_name)
    for frame in _sort_frames(frame_dir):
        print(f"- processing {frame.name}...")
        frame_num = _get_frame_number(frame)
        current_frame = _describe_frame(get_response, frame_num, frame, questions)
        images.append(current_frame)

    fps = get_frame_rate(video_path)

    answer = {
        'video_name': video_path.name,
        'fps': fps,
        'frames': images
    }

    return answer
        

def process_videos(video_dir: Path,
                   frame_dir: Path,
                   frame_step_size: int = FRAME_STEP_SIZE,
                   questions: list[str] = [],
                   model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
                   output_file: Path | None = None):
                                      
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

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(answers, f, indent=4) 

    return answers


if __name__ == "__main__":
    prompts = VLM_PROMPTS
    video_dir = LLM_VIDEO_DIR
    frame_dir = LLM_FRAME_DIR

    models = ["llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llava-1.5-7b-hf", "Salesforce/blip2-opt-2.7b"]

    # start = time.time()    
    # model_name = "liuhaotian/llava-v1.5-7b"
    # output_file = LLM_DATA_DIR / "blip2-opt-2.7b.json"
    # process_videos(video_dir, frame_dir, 20, prompts, model_name, output_file)
    # end = time.time()   
    # print(f"\n{round(end - start, 2)} seconds elapsed...")

    for model_name in models: 
        start = time.time()
        model_json_file = model_name.split("/")[1].replace(".", "-")
        output_file = LLM_DATA_DIR / f"{model_json_file}.json"
        process_videos(video_dir, frame_dir, 20, prompts, model_name, output_file)
        end = time.time()   
        print(f"\n{round(end - start, 2)} seconds elapsed...")