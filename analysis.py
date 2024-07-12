"""This file uses data from data.json"""
import json
import spacy
import subprocess
import pandas as pd
import numpy as np
import pprint
from pathlib import Path
from typing import List, Dict
import gensim.downloader
import pathlib as Path
from gensim.models import KeyedVectors
import yaml

with open("data.json", 'r') as f:
    data = json.load(f)

# load small spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # download the english model if it doesn't exist already
    subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


INGREDIENT_PROMPT_INDEX = 3

# prompt index is the question number in the list of prompts that corresponds to the query you are interested in
def parse_comma_list(list_str: str) -> list:
    """Parses a comma-separated list (list can include 'and') and extracts individual items.
    
    Filtering/cleaning criteria include:
    - entries must be a noun
    - singular, plural forms of the same entry are counted as a single entry
    - duplicate entries are removed
    """
    # # check if video exists, collect data
    # video_data = None
    # for video in data:
    #     if video_name == video['video_name']:
    #         video_data = video['frames']

    # # if video doesn't exist, raise an error
    # if not video_data:
    #     raise ValueError(f"{video_name} video not found")

    # ingredients = set()
    # for frame in video_data:
    #     frame_ingredients = frame['questions'][prompt_index]['answer']

    # clean up ingredients list
    for word in ['and', '.']: # DO NOT PUT ',' here
        list_str = list_str.replace(word, '')
    processed_ingredients = [item.lower().strip() for item in list_str.split(', ')]

    # union this with existing ingredients set
    ingredients = set(processed_ingredients)

    # remove words that represent the same item, just in different quantities
    singular_items = set()

    for item in ingredients:
        # converting all nouns to singular form
        doc = nlp(item)
        singular_nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]

        # adding each token to set of singular items
        word = " ".join(singular_nouns)
        singular_items.add(word)

    # remove empty strings, single characters
    answer = filter(lambda x: len(x) > 1, singular_items)
    answer = sorted(answer)

    return answer


def match_outputs(llm_lst: List[str], gt_lst: List[str]):
    """
    Matches outputs of a list from an LLM to a ground truth list.
    
    Whenever an identical entry is found in either, drops both entries from both lists. 
    
    If the `gt_lst` contains an entry that allows for multiple variations of a word, then that word is dropped from the `llm_lst` and not from the `gt_lst` in case of any future matches.
    
    Finally, after processing all elements, if a `gt_lst` has been matched before, it is dropped too.

    Returns a diff between both lists: a dictionary
    """
    gt_edit_lst = gt_lst.copy()
    llm_edit_lst = llm_lst.copy()

    # need to ensure all words in a single multiple match key are in lowercase form
    multiple_matches_used = {}    
    true_positive = []

    for word in gt_lst:
        if any(char in word for char in ['|', ',']):
            multiple_matches_used[word] = False

    for word in llm_lst:
        if word in gt_lst:
            gt_edit_lst.remove(word)
            llm_edit_lst.remove(word)
            true_positive.append(word)
        else:
            for key, value in multiple_matches_used.items():
                if word in key:
                    multiple_matches_used[key] = True
                    llm_edit_lst.remove(word)
                    true_positive.append(word)
                    # no gt_edit_lst here since we keep multiple matches even if we have a single match alr
    
    for key, value in multiple_matches_used.items():
        if value:
            gt_edit_lst.remove(key)
        else:
            individual_items = [item.strip() for item in key.split('||')]
            gt_edit_lst.remove(key)
            # gt_edit_lst += individual_items
            gt_edit_lst.append(individual_items[0])

    # diff = {'LLM': [], 'Ground Truth': []}
    diff = {'False Positive': [], 'False Negative': [], 'True Positive': []}

    for word in llm_edit_lst:
        diff['False Positive'].append(word)
    for word in gt_edit_lst:
        diff['False Negative'].append(word)
    for word in true_positive:
        diff['True Positive'].append(word)

    return diff


def compute_diff_score(diff_dict: Dict[str, List[str]]):
    try:
        model = KeyedVectors.load("word2vec/word2vec.model")
    except:
        print("Downloading Word2Vec model...")
        glove = gensim.downloader.load('glove-wiki-gigaword-300')

        model_path = Path('word2vec/')
        model_path.mkdir(parents=True)

        glove.save('word2vec/word2vec.model')
        model = KeyedVectors.load("word2vec/word2vec.model")

    dists = []
    
    # step 1
    for llm_word in diff_dict['False Positive']:
        cosine_dist = []
        for gt_word in diff_dict['False Negative']:
            try:
                dist = model.distance(llm_word, gt_word)
                cosine_dist.append(dist)
            except KeyError:
                print(f"Word2Vec does not contain '{gt_word}'")
                continue
        if len(cosine_dist) != 0:
            dists.append(min(cosine_dist))

    # step 2
    for gt_word in diff_dict['False Negative']:
        cosine_dist = []
        for llm_word in diff_dict['False Positive']:
            try:
                dist = model.distance(gt_word, llm_word)
                cosine_dist.append(dist)
            except KeyError:
                print(f"Word2Vec does not contain '{gt_word}'")
                continue
        if len(cosine_dist) != 0:
            dists.append(min(cosine_dist))

    return sum(dists)


def parse_yaml(config_path: str):
    config_path = Path(config_path)
    if not config_path.exists():
        raise ValueError(f"File {config_path} doesn't exist")

    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)

    parsed_data = {
        'video_name': data['video_name'],
        'frames': []
    }

    for frame in data['frames']:
        parsed_frame = {
            'number': frame['number'],
            'utensils': frame['utensils'],
            'ingredients': frame['ingredients']
        }
        parsed_data['frames'].append(parsed_frame)

    return parsed_data
   

def get_video_frame_data(video_name: str, frame_number: int, prompt_index: int):
    with open('data.json', 'r') as f:
        video_data = json.load(f)

    for video in video_data:
        if video_name == video['video_name']:
            for frame in video['frames']:
                if frame['frame_number'] == frame_number:
                    data_list = frame['questions'][prompt_index]['answer']
                    return parse_comma_list(data_list)

            raise ValueError(f"Provided video name does not have any data on frame #{frame_number}") 

    raise ValueError(f"Provided video name {video_name} does not exist in data.json")


def compare_pred(video_name: str, frame_number: int):
    # config_file = Path(f"config-custom-videos/{video_name}.yaml")
   
    # if not config_file.exists():
    #     raise ValueError(f"Provided video name {video_name} does not have a corresponding yaml file")
    
    yaml_data = parse_yaml(f"config-custom-videos/{video_name}.yaml")
    frame_data = yaml_data['frames'] 

    gt_lst = None
    for frame in frame_data:
        if frame['number'] == frame_number:
            gt_lst = frame['ingredients']

    if not gt_lst:
        raise ValueError(f"Provided frame number does not exist in config file")
    
    llm_lst = get_video_frame_data(video_name, frame_number, INGREDIENT_PROMPT_INDEX)

    diff = match_outputs(llm_lst, gt_lst)
    score = compute_diff_score(diff)
    
    return diff, score


if __name__ == "__main__":
    # test #1
    # llm_lst = ["omelette", "egg", "cucumber", "fries", "potato", "tomato"]
    # gt_lst = ["carrot", "egg || omelette", "tomatoes || tomato", "potato || fries || potato fries"]

    # diff = match_outputs(llm_lst, gt_lst)
    # score = compute_diff_score(diff)

    # print(f"Score: {score}, Diff: {diff}")


    # # test #2
    # llm_lst = ["spaghetti", "pasta", "meatballs", "tomato sauce", "cheese"]
    # gt_lst = ["spaghetti || pasta", "meatball || meatballs", "tomato sauce || marinara", "cheese || parmesan"]

    # diff = match_outputs(llm_lst, gt_lst)
    # score = compute_diff_score(diff)

    # print(f"Score: {score}, Diff: {diff}")


    # # test #3
    # llm_lst = ["pasta", "alfredo sauce", "chicken", "broccoli", "parmesan"]
    # gt_lst = ["pasta || noodles", "alfredo sauce || white sauce", "chicken || turkey", "broccoli || peas", "parmesan || cheddar"]

    # diff = match_outputs(llm_lst, gt_lst)
    # score = compute_diff_score(diff)

    # print(f"Score: {score}, Diff: {diff}")


    # # test #4
    # llm_lst = ["hamburger", "lettuce", "tomato", "pickles", "bun"]
    # gt_lst = ["spinach", "pickles || relish", "bread roll"]

    # diff = match_outputs(llm_lst, gt_lst)
    # score = compute_diff_score(diff)

    # print(f"Score: {score}, Diff: {diff}")
    
    # mp4_0_yaml = parse_yaml("config-custom-videos/0.mp4.yaml")
    # pprint.pprint(mp4_0_yaml)

    # data = get_video_frame_data(
    #     video_name='0.mp4', 
    #     frame_number=10, 
    #     prompt_index=INGREDIENT_PROMPT_INDEX
    # )
    # print(data)

    diff, score = compare_pred(video_name='2.mp4', frame_number=15)
    print(f"Score: {score}, Diff: {diff}")