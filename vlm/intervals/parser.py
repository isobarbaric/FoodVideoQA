import yaml
import spacy
import subprocess
from pathlib import Path
from hyperparameters import UTENSILS
from typing import List, Dict

# load small spaCy English model: https://spacy.io/models/en
def load_spacy_model() -> spacy.Language:
    """
    Load the spaCy English language model. If the model is not available, download it.

    Returns:
        spacy.Language: The spaCy language model.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_spacy_model()


def parse_comma_list(list_str: str) -> List[str]:
    """
    Parse a comma-separated list of items, clean it, and extract unique nouns in their singular form.

    Args:
        list_str (str): The comma-separated list as a string.

    Returns:
        List[str]: A sorted list of unique noun items in singular form.
    """

    # clean up ingredients list
    for word in ['and', '.']:
        list_str = list_str.replace(word, '')

    items = [item.lower().strip() for item in list_str.split(', ')]

    # remove words that represent the same item, just in different quantities
    singular_items = set()

    for item in items:
        # extracting the actual food item from the list
        doc = nlp(item)
        singular_nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]

        # adding each token to set of singular items
        if len(singular_nouns) != 0:
            singular_items.add(' '.join(singular_nouns))

    # remove empty strings, single characters
    return list(singular_items)

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