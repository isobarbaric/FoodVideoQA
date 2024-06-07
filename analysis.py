"""This file uses data from data.json"""
import json
import spacy
import subprocess
import pandas as pd
import pprint

INGREDIENT_PROMPT_INDEX = 2

with open("data.json", 'r') as f:
    data = json.load(f)

# load small spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # download the english model if it doesn't exist already
    subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def extract_weights():
    with open("data.json", 'r') as file:
        data = json.load(file)

    weight_map = {}

    for video in data:
        video_name = video["video_name"]
        weights = []

        for frame in video["frames"]:
            for question in frame["questions"]:
                if "weight" in question["prompt"]:
                    weight = int(question["answer"].split()[0].replace(',', ''))
                    weights.append(weight)
        
        weight_map[video_name] = weights

    return weight_map


def video_ingredients(video_name: str) -> list:
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
        frame_ingredients = frame['questions'][INGREDIENT_PROMPT_INDEX]['answer']

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


def print_weights_ingredients(weight_map):
    for video in weight_map.keys():
        ingredients = video_ingredients(video)
        print(video)
        print("avg weight: ", round(sum(weight_map[video])/len(weight_map[video]), 2), "grams")
        print("weights:", weight_map[video])
        print("ingredients:", ingredients)
        print()


def create_df():
    data = []
    for video in weight_map.keys():
        rn = dict()
        ingredients = video_ingredients(video)
        rn['video_name'] = video
        rn['avg_weight'] = round(sum(weight_map[video])/len(weight_map[video]), 2) # this is in grams
        rn['weights'] = weight_map[video]
        rn['ingredients'] = ingredients

        data.append(rn)
    
    return pd.DataFrame(data)

   
if __name__ == "__main__":
    weight_map = extract_weights()
    # print_weights_ingredients(weight_map)

    df = create_df()
    print(df)