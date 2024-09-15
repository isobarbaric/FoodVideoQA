import yaml
import spacy
import subprocess
from pathlib import Path
from llm.generation.generate import UTENSILS
from typing import List, Dict


# load small spaCy English model
def load_spacy_model() -> spacy.Language:
    """
    Load the spaCy English language model. If the model is not available, download it.

    Returns:
        spacy.Language: The spaCy language model.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Download the English model if it doesn't exist
        subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_spacy_model()


# prompt index is the question number in the list of prompts that corresponds to the query you are interested in
def parse_comma_list(list_str: str) -> List[str]:
    """
    Parse a comma-separated list of items, clean it, and extract unique nouns in their singular form.

    Args:
        list_str (str): The comma-separated list as a string.

    Returns:
        List[str]: A sorted list of unique noun items in singular form.
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


def parse_utensils_list(utensil_response: str) -> List[str]:
    """
    Parse a response containing utensils, clean the response, and filter out valid utensils.

    Args:
        utensil_response (str): The response string containing utensils.

    Returns:
        List[str]: A list of validated utensils in lowercase.
    """
    for word in [',', '-', '.',]:
        utensil_response = utensil_response.replace(word, '')    
    print(utensil_response)

    cleaned_utensils = []
    for word in utensil_response.split(' '):
        if word.lower() in UTENSILS:
            cleaned_utensils.append(word.lower())
    return cleaned_utensils


def parse_yaml(config_path: str) -> Dict:
    """
    Parse a YAML configuration file and extract relevant data.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict: A dictionary containing parsed video and frame data.

    Raises:
        ValueError: If the configuration file does not exist.
    """
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