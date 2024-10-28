import json
from pathlib import Path
from vlm.intervals.parser import parse_comma_list
import pprint

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
LLM_DATA_DIR = DATA_DIR / 'llm'

with open(LLM_DATA_DIR / 'temp_test_data.json') as f:
    video_data = json.load(f)

food_item_idx = 0
food_data = []

for video in video_data:
    curr = []
    for frames in video['frames']:
        curr.append(frames['questions'][food_item_idx]['answer'])
    curr = [parse_comma_list(answer) for answer in curr]
    food_data.append(curr)
    break # only looking at the first video for now

print(f'# of videos: {len(food_data)}')
pprint.pprint(food_data[0][:20])