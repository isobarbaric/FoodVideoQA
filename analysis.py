"""This file uses data from data.json"""
import json
import spacy
import subprocess
import pandas as pd
import pprint
from pathlib import Path

with open("data.json", 'r') as f:
    data = json.load(f)

# load small spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # download the english model if it doesn't exist already
    subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# prompt index is the question number in the list of prompts that corresponds to the query you are interested in
def parse_comma_list(video_name: str, prompt_index: list) -> list:
    """Parses a comma-separated list (list can include 'and') and extracts individual items.
    
    Filtering/cleaning criteria include:
    - entries must be a noun
    - singular, plural forms of the same entry are counted as a single entry
    - duplicate entries are removed
    """
    # check if video exists, collect data
    video_data = None
    for video in data:
        if video_name == video['video_name']:
            video_data = video['frames']

    # if video doesn't exist, raise an error
    if not video_data:
        raise ValueError(f"{video_name} video not found")

    ingredients = set()
    for frame in video_data:
        frame_ingredients = frame['questions'][prompt_index]['answer']

        # clean up ingredients list
        for word in ['and', '.']: # DO NOT PUT ',' here
            frame_ingredients = frame_ingredients.replace(word, '')
        processed_ingredients = [item.lower().strip() for item in frame_ingredients.split(', ')]

        # union this with existing ingredients set
        ingredients |= set(processed_ingredients)

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

   
if __name__ == "__main__":
    pass