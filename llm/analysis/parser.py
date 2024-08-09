import yaml
import spacy
import subprocess
from pathlib import Path
from llm.generation.generate import utensils


# load small spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # download the english model if it doesn't exist already
    subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# prompt index is the question number in the list of prompts that corresponds to the query you are interested in
def parse_comma_list(list_str: str) -> list:
    """Parses a comma-separated list (list can include 'and') and extracts individual items.
    
    Filtering/cleaning criteria include:
    - entries must be a noun
    - singular, plural forms of the same entry are counted as a single entry
    - duplicate entries are removed
    """
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


def parse_utensils_list(utensil_response: str):
    for word in [',', '-', '.',]:
        utensil_response = utensil_response.replace(word, '')    
    print(utensil_response)

    cleaned_utensils = []
    for word in utensil_response.split(' '):
        if word.lower() in utensils:
            cleaned_utensils.append(word.lower())
    return cleaned_utensils


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