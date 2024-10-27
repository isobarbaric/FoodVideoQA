# word-to-word matching
# semantic matching (using Word2Vec)

import json
import pprint
from pathlib import Path
from typing import List, Dict
from pathlib import Path
from llm.analysis.parser import parse_comma_list, parse_yaml, parse_utensils_list
from llm.analysis.scoring import match_outputs, compute_diff_score
from rich.console import Console

# directories
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LLM_DATA_DIR = DATA_DIR / "llm"
JSON_DATA_PATH = LLM_DATA_DIR / 'data.json'
VIDEO_ANALYSIS_DIR = LLM_DATA_DIR / "video-analysis"


# constants: defined as written in the data.json file
UTENSILS_EATING_INDEX = 1
UTENSILS_PROMPT_INDEX = 2
INGREDIENTS_PROMPT_INDEX = 3

console = Console()


def get_video_frame_data(video_name: str, frame_number: int, prompt_index: int):
    """
    Retrieve and parse data for a specific frame and prompt from the video data.

    Args:
        video_name (str): The name of the video containing the frame.
        frame_number (int): The number of the frame to retrieve data for.
        prompt_index (int): Index indicating which prompt's data to retrieve.

    Returns:
        list: A parsed list of data corresponding to the specified prompt index.

    Raises:
        ValueError: If the video name or frame number does not exist in the data, or if the prompt_index is invalid.
    """
    with open(JSON_DATA_PATH, 'r') as f:
        video_data = json.load(f)


    video_name = "video_" + video_name

    for video in video_data:
        if video_name == video['video_name']:
            for frame in video['frames']:
                if frame['frame_number'] == frame_number:
                    data_list = frame['questions'][prompt_index]['answer']
                    if prompt_index == INGREDIENTS_PROMPT_INDEX:
                        return parse_comma_list(data_list)
                    elif prompt_index == UTENSILS_PROMPT_INDEX:
                        return parse_utensils_list(data_list)
                    elif prompt_index == UTENSILS_EATING_INDEX:
                        return parse_utensils_list(data_list)

            raise ValueError(f"Provided video name does not have any data on frame number: {frame_number}") 

    raise ValueError(f"Provided video name {video_name} does not exist in data.json")


def compare_pred(video_name: str, frame_number: int, prompt_index: int):
    """
    Compare the predictions for a specific frame against ground truth data.

    Args:
        video_name (str): The name of the video for which to compare predictions.
        frame_number (int): The number of the frame to compare.
        prompt_index (int): Index indicating which prompt's data to compare.

    Returns:
        tuple: A tuple containing:
            - A list of differences between the predicted and ground truth data.
            - A computed score indicating the accuracy of the predictions.

    Raises:
        ValueError: If the frame number does not exist in the configuration file or if the prompt_index is invalid.
    """
    yaml_data = parse_yaml(VIDEO_ANALYSIS_DIR / f"{video_name}.yaml")
    frame_data = yaml_data['frames'] 

    gt_lst = None
    for frame in frame_data:
        if frame['number'] == frame_number:
            gt_lst = frame['ingredients']

    if not gt_lst:
        raise ValueError(f"Provided frame number does not exist in config file")
    
    SUPPORTED_PROMPT_INDICES = [INGREDIENTS_PROMPT_INDEX, UTENSILS_EATING_INDEX]
    if prompt_index not in SUPPORTED_PROMPT_INDICES:
        raise ValueError(f"Invalid value for prompt_index; prompt index must be in {SUPPORTED_PROMPT_INDICES}")

    # prompt_index can be either INGREDIENTS_PROMPT_INDEX or UTENSILS_PROMPT_INDEX
    llm_lst = get_video_frame_data(video_name, frame_number, prompt_index)
    # print()
    # console.print(f"LLM list: {llm_lst}", style="yellow")
    # console.print(f"GT list: {gt_lst}", style="yellow")

    diff = match_outputs(llm_lst, gt_lst)
    score = compute_diff_score(diff)
    
    return diff, score


if __name__ == "__main__":
    """
    test #1
    llm_lst = ["omelette", "egg", "cucumber", "fries", "potato", "tomato"]
    gt_lst = ["carrot", "egg || omelette", "tomatoes || tomato", "potato || fries || potato fries"]

    diff = match_outputs(llm_lst, gt_lst)
    score = compute_diff_score(diff)

    print(f"Score: {score}, Diff: {diff}")
    """


    """
    test #2
    llm_lst = ["spaghetti", "pasta", "meatballs", "tomato sauce", "cheese"]
    gt_lst = ["spaghetti || pasta", "meatball || meatballs", "tomato sauce || marinara", "cheese || parmesan"]

    diff = match_outputs(llm_lst, gt_lst)
    score = compute_diff_score(diff)

    print(f"Score: {score}, Diff: {diff}")
    """


    """
    test #3
    llm_lst = ["pasta", "alfredo sauce", "chicken", "broccoli", "parmesan"]
    gt_lst = ["pasta || noodles", "alfredo sauce || white sauce", "chicken || turkey", "broccoli || peas", "parmesan || cheddar"]

    diff = match_outputs(llm_lst, gt_lst)
    score = compute_diff_score(diff)

    print(f"Score: {score}, Diff: {diff}")
    """


    """
    test #4
    llm_lst = ["hamburger", "lettuce", "tomato", "pickles", "bun"]
    gt_lst = ["spinach", "pickles || relish", "bread roll"]

    diff = match_outputs(llm_lst, gt_lst)
    score = compute_diff_score(diff)

    print(f"Score: {score}, Diff: {diff}")
    """
    

    """
    mp4_0_yaml = parse_yaml("config-custom-videos/0.mp4.yaml")
    pprint.pprint(mp4_0_yaml)

    data = get_video_frame_data(
        video_name='0.mp4', 
        frame_number=10, 
        prompt_index=INGREDIENT_PROMPT_INDEX
    )
    print(data)
    """

    # diff, score = compare_pred(video_name='2.mp4', frame_number=15, prompt_index=INGREDIENTS_PROMPT_INDEX)
    # print(f"Score: {score}, Diff: {diff}")