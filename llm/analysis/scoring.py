from pathlib import Path
from gensim.models import KeyedVectors
import gensim.downloader
import numpy as np


def match_outputs(llm_lst: list[str], gt_lst: list[str]) -> dict:
    """
    Matches the outputs from a list of LLM predictions to a ground truth list. 

    Identical entries in both lists are considered true positives and removed from both. If the ground truth list contains entries
    that allow for multiple variations, only one of those variations is removed from the LLM list.

    Args:
        llm_lst (list[str]): List of predictions from the LLM.
        gt_lst (list[str]): List of ground truth values.

    Returns:
        dict: A dictionary containing lists of 'False Positive', 'False Negative', and 'True Positive' items.
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
    
    for key, value in multiple_matches_used.items():
        if value:
            gt_edit_lst.remove(key)
        else:
            individual_items = [item.strip() for item in key.split('||')]
            gt_edit_lst.remove(key)
            gt_edit_lst.append(individual_items[0])

    diff = {
        'False Positive': llm_edit_lst,
        'False Negative': gt_edit_lst,
        'True Positive': true_positive
    }

    return diff


def compute_diff_score(diff_dict: dict[str, list[str]]) -> float:
    """
    Computes the similarity score between 'False Positive' and 'False Negative' entries using Word2Vec.

    Args:
        diff_dict (dict[str, list[str]]): Dictionary containing 'False Positive' and 'False Negative' lists.

    Returns:
        float: The computed similarity score based on Word2Vec distances.
    """
    try:
        model = KeyedVectors.load("word2vec/word2vec.model")
    except:
        print("Downloading Word2Vec model...")
        glove = gensim.downloader.load('glove-wiki-gigaword-300')

        model_path = Path('word2vec/')
        model_path.mkdir(parents=True)

        glove.save('word2vec/word2vec.model')
        model = KeyedVectors.load("word2vec/word2vec.model")

    false_positive = [word for word in diff_dict['False Positive'] if word in model]
    false_negative = [word for word in diff_dict['False Negative'] if word in model]
    dists = []
    
    # Step 1: Find the closest word in 'False Negative' for each word in 'False Positive'
    for llm_word in false_positive:
        if not false_negative:
            continue
        cosine_dists = model.distances(llm_word, false_negative)
        dists.append(np.min(cosine_dists))

    # Step 2: Find the closest word in 'False Positive' for each word in 'False Negative'
    for gt_word in false_negative:
        if not false_positive:
            continue
        cosine_dists = model.distances(gt_word, false_positive)
        dists.append(np.min(cosine_dists))

    return sum(dists)