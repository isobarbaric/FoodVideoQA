from pathlib import Path
import numpy as np
from bert_score import BERTScorer, plot_example
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from rich.console import Console
from vlm.analysis.matching import semantic_matching, word_matching
import json

console = Console()
scorer = BERTScorer(model_type='bert-large-uncased', lang="en", rescale_with_baseline=True)

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LLM_DATA_DIR = DATA_DIR / "llm"

# ground truth data
DATASET_DIR = ROOT_DIR / "dataset"
GT_JSON = DATASET_DIR / "ground_truth.json"

# VLM-generated data
BLIP2_JSON = LLM_DATA_DIR / "blip2-opt-2-7b.json"
LLAVA_1_5_JSON = LLM_DATA_DIR / "llava-1-5-7b-hf.json"
LLAVA_1_6_JSON = LLM_DATA_DIR / "llava-v1-6-mistral-7b-hf.json"


def load_json(file: Path) -> dict:
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def get_ground_truth() -> dict:
    return load_json(GT_JSON)

def get_average_BERT_score(llm_data: dict, gt_data: dict) -> tuple[float, float, float]:

    num_videos = len(gt_data)
    P = 0
    R = 0
    F1 = 0
    iterations = 0

    for video_index in range(num_videos):
        llm_frames_lst = llm_data[video_index]['frames']
        gt_frames_lst = gt_data[video_index]['frames']


        llm_foods_lst = [list(frame['questions'][0]['answer'].split(" ")) for frame in llm_frames_lst]
        gt_foods_lst = [ list(frame['info']['food items']) for frame in gt_frames_lst]
        assert(len(llm_foods_lst) == len(gt_foods_lst))
        for llm_foods, gt_foods in zip(llm_foods_lst, gt_foods_lst):
            P_1, R_1, F1_1 = semantic_matching(llm_foods, gt_foods)
            iterations += 1
            P += P_1
            R += R_1
            F1 += F1_1

        llm_utensils_lst = [list(frame['questions'][1]['answer'].split(" ")) for frame in llm_frames_lst]
        gt_utensils_lst = [list(frame['info']['utensils']) for frame in gt_frames_lst]
        assert(len(llm_utensils_lst) == len(gt_utensils_lst))
        for llm_utensils, gt_utensils in zip(llm_utensils_lst, gt_utensils_lst):
            P_2, R_2, F1_2 = semantic_matching(llm_utensils, gt_utensils)
            iterations += 1
            P += P_2
            R += R_2
            F1 += F1_2
        
        llm_ingred_lst = [list(frame['questions'][2]['answer'].split(" ")) for frame in llm_frames_lst]
        gt_ingred_lst = [list(frame['info']['ingredients']) for frame in gt_frames_lst]
        assert(len(llm_ingred_lst) == len(gt_ingred_lst))
        for llm_ingred, gt_ingred in zip(llm_ingred_lst, gt_ingred_lst):
            P_3, R_3, F1_3 = semantic_matching(llm_ingred, gt_ingred)
            iterations += 1
            P += P_3
            R += R_3
            F1 += F1_3

    P /= iterations
    R /= iterations
    F1 /= iterations

    return P, R, F1



if __name__ == "__main__":
    gt_data = get_ground_truth()
    blip2_data = load_json(BLIP2_JSON)
    llava_1_5_data = load_json(LLAVA_1_5_JSON)
    llava_1_6_data = load_json(LLAVA_1_6_JSON)

    P_blip2, R_blip2, F1_blip2 = get_average_BERT_score(blip2_data, gt_data)
    P_llava_1_5, R_llava_1_5, F1_llava_1_5 = get_average_BERT_score(llava_1_5_data, gt_data)
    P_llava_1_6, R_llava_1_6, F1_llava_1_6 = get_average_BERT_score(llava_1_6_data, gt_data)

    console.print(f"BLIP2: P: {P_blip2}, R: {R_blip2}, F1: {F1_blip2}", style="bold cyan")
    console.print(f"LLAVA1.5: P: {P_llava_1_5}, R: {R_llava_1_5}, F1: {F1_llava_1_5}", style="bold magenta")
    console.print(f"LLAVA1.6: P: {P_llava_1_6}, R: {R_llava_1_6}, F1: {F1_llava_1_6}", style="bold green")
