from pathlib import Path
from gensim.models import KeyedVectors
import gensim.downloader


def match_outputs(llm_lst: list[str], gt_lst: list[str]):
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


def compute_diff_score(diff_dict: dict[str, list[str]]):
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
