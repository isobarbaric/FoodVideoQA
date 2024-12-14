import json
from pathlib import Path
from vlm.intervals.parser import parse_comma_list
from hyperparameters import FRAME_STEP_SIZE
import pprint

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
LLM_DATA_DIR = DATA_DIR / 'llm'
FOOD_ITEM_IDX = 0

with open(LLM_DATA_DIR / 'data.json') as f:
    video_data = json.load(f)

video_fps = dict()

# parse VLM output into individual food items
food_data = []
for video in video_data:
    curr = []
    for frames in video['frames']:
        curr.append(frames['questions'][FOOD_ITEM_IDX]['answer'])
    curr = [parse_comma_list(answer) for answer in curr]
    video_fps[video['video_name']] = video['fps']
    food_data.append(curr)

# defines when the current interval ends
def next_idx(vals: list[list[str]], repeat_val: str, starting_idx: int) -> int:
    for i in range(starting_idx, len(vals)):
        if repeat_val not in vals[i]:
            return i
    return len(vals)

# stitches together intervals of consistent eating of a particular food item from a single video's data (as in data.json)
def create_intervals_naive(food_data_video: list[list[str]], video_name: str) -> list[list[int]]:
    intervals = []
    curr_idx = 0
    while curr_idx < len(food_data_video):
        if len(food_data_video[curr_idx]) == 0:
            curr_idx += 1
            continue
        
        curr_best = []
        for food_str in food_data_video[curr_idx]:
            food_next_idx = next_idx(food_data_video, food_str, curr_idx+1)
            
            # require that the interval isn't just a single value (atleast 2 consecutive occurences are needed)
            if food_next_idx != curr_idx+1:
                curr_best.append([food_next_idx-curr_idx, [curr_idx, food_next_idx, food_str]])

        # sort by length greedily to grab the longest interval that can be made
        curr_best.sort(reverse=True)

        if len(curr_best) > 0:
            intervals.append(curr_best[0][1])
            food_next_idx = curr_best[0][1][1]
            # set curr_idx to the last idx of the interval just merged
            curr_idx = food_next_idx
        else:
            curr_idx += 1
        
    for interval in intervals:
        interval[0] = round(interval[0] / video_fps[video_name] * FRAME_STEP_SIZE, 2)
        interval[1] = round(interval[1] / video_fps[video_name] * FRAME_STEP_SIZE, 2)
                                         
    return intervals


# tolerance for how many empty frames between frames of the same food
TOLERANCE = 15
SECONDS_TOLERANCE = 0.1

def create_intervals_optimized(food_data_video: list[list[str]], video_name: str) -> list[list[int]]:
    intervals = []
    curr_idx = 0
    while curr_idx < len(food_data_video):
        if len(food_data_video[curr_idx]) == 0:
            curr_idx += 1
            continue
        
        curr_best = []
        for food_str in food_data_video[curr_idx]:
            current_ending = next_idx(food_data_video, food_str, curr_idx+1)

            done = False            
            # i is all possible starting points
            for i in range(current_ending+TOLERANCE, len(food_data_video)):
                if len(food_data_video[i]) == 0:
                    continue
                if food_data_video[i] != food_str:
                    continue
                # so now, 'i' contains the same food
                curr_next_idx = next_idx(food_data_video, food_str, i+1)
                
                # require that the interval isn't just a single value (atleast 2 consecutive occurences are needed)
                if food_next_idx != curr_idx+1:
                    curr_best.append([current_ending-curr_idx, curr_next_idx-current_ending, [curr_idx, current_ending, food_str]])
                    done = True

            if not done:
                curr_best.append([current_ending-curr_idx, 0, [curr_idx, current_ending, food_str]])

        # sort by length greedily to grab the longest interval that can be made
        curr_best.sort(reverse=True)

        if len(curr_best) > 0:
            intervals.append(curr_best[0][2])
            food_next_idx = curr_best[0][2][1]
            # set curr_idx to the last idx of the interval just merged
            curr_idx = food_next_idx
        else:
            curr_idx += 1
    
    ret_intervals = []
    for i in range(len(intervals)-1):
        interval = intervals[i]
        interval[0] = round(interval[0] / video_fps[video_name] * FRAME_STEP_SIZE, 2)
        interval[1] = round(interval[1] / video_fps[video_name] * FRAME_STEP_SIZE, 2)

        # if interval[1] - interval[0] >= SECONDS_TOLERANCE:
        #     ret_intervals.append(interval)
        ret_intervals.append(interval)

    return ret_intervals


# TODO: deal with edge case with last elemtn
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    merged_intervals = []
    # these are guaranteed to be sorted, and within TOLERANCE
    curr_idx = 0
    while curr_idx < len(intervals)-1:
        if len(merged_intervals) == 0 or intervals[curr_idx][2] != intervals[curr_idx+1][2]:
            merged_intervals.append(intervals[curr_idx])
            curr_idx += 1
        else:
            food_str = intervals[curr_idx][2]
            merged_intervals.append([intervals[curr_idx][0], intervals[curr_idx+1][1], food_str])
            curr_idx += 2
    return merged_intervals


if __name__ == "__main__":
    videos = []
    for video_idx in range(len(food_data)):
        video_name = video_data[video_idx]['video_name']
        # curr_video_intervals = create_intervals_naive(food_data[video_idx], video_name)
        curr_video_intervals = create_intervals_optimized(food_data[video_idx], video_name)
        videos.append(curr_video_intervals)
    print("Intervals:")
    pprint.pprint(videos)

    print("Merged Intervals:")
    for i, video_intervals in enumerate(videos):
        merged_interval = merge_intervals(video_intervals)
        print(f"Video {i+1}:")
        pprint.pprint(merged_interval)