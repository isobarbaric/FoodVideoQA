from pathlib import Path
from rich.console import Console
import pprint
import time
import json
import os

from hyperparameters import VLM_PROMPTS, FOOD_ITEM_IDX, FRAME_STEP_SIZE
from vlm.generation.generate import process_videos
from pose.eat import determine_eating, make_get_bounding_boxes
from vlm.intervals.stitcher import parse_comma_list, create_intervals_optimized, merge_intervals, annotate_and_save_frames, create_video_from_frames


# Input: MP4 video
# Step 1: Slice into frames
# Step 2: Run VLM => data.json
# Step 3: Run Pose => data.json (eating vs not eating)
# Step 4: Run Intervals => output video
# Output: MP4 video with annotations


DATA_DIR = Path('data_inference')

VLM_DATA_DIR = DATA_DIR / 'vlm'
INPUT_VIDEOS_DIR = VLM_DATA_DIR / 'videos'
VLM_FRAMES_DIR = VLM_DATA_DIR / 'frames'

POSE_DATA_DIR = DATA_DIR / 'pose'
POSE_ANNOTATED_FRAMES_DIR = POSE_DATA_DIR / 'annotated_frames'
POSE_FACE_PLOT_DIR = POSE_DATA_DIR / 'face_plots'

INTERVALS_DATA_DIR = DATA_DIR / 'intervals'
INTERVALS_FRAMES_DIR = INTERVALS_DATA_DIR / 'output_frames'
OUTPUT_VIDEOS_DIR = INTERVALS_DATA_DIR / 'output_videos'

DATA_JSON = DATA_DIR / 'data.json'

VLM_MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
POSE_MODEL_NAME = "IDEA-Research/grounding-dino-base"

console = Console()

def generate_vlm_insights(model_name=VLM_MODEL_NAME, output_file=DATA_JSON, video_dir=INPUT_VIDEOS_DIR, frame_dir=VLM_FRAMES_DIR, prompts=VLM_PROMPTS):
    start = time.time()
    process_videos(video_dir, frame_dir, 20, prompts, model_name, output_file)
    end = time.time()   
    print(f"\n{round(end - start, 2)} seconds elapsed...")



def generate_pose_annotations(model_name=POSE_MODEL_NAME, input_image_dir=VLM_FRAMES_DIR, output_image_dir=POSE_ANNOTATED_FRAMES_DIR, output_face_plot_dir=POSE_FACE_PLOT_DIR, output_file=DATA_JSON):
    generate_bounding_boxes = make_get_bounding_boxes(model_name)
    
    with open(DATA_JSON, "r") as file:
        data = json.load(file)
    
    for video_index, video_dir in enumerate(sorted(input_image_dir.iterdir())):
        if not video_dir.is_dir():
            continue
    
        video_name = f"video_{video_index + 1}.mp4"
        video_data = next((v for v in data if v["video_name"] == video_name), None)
        if not video_data:
            console.print(f"[red]Video {video_name} not found in data.json[/red]")
            continue

        for frame_data in video_data["frames"]:
            frame_number = frame_data["frame_number"]
            frame_name = f"frame{frame_number}"

            console.print(f"[green]Processing[/green] {video_name} {frame_name}")

            image_path = video_dir / f"{frame_name}.jpg"
            output_path = output_image_dir / video_name / f"{frame_name}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            face_plot_output = output_face_plot_dir / video_name / f"{frame_name}.jpg"
            face_plot_output.parent.mkdir(parents=True, exist_ok=True)

            eating = determine_eating(generate_bounding_boxes, image_path, output_path, face_plot_output_path=face_plot_output, mouth_bbox_output_path=None)
            frame_data["eating"] = 1 if eating else 0
            console.print(f"[green]Eating:[/green] {eating}")

    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)




def generate_intervals(output_file=DATA_JSON, food_item_prompt_index=FOOD_ITEM_IDX, include_bboxes=True, output_frames_dir=INTERVALS_FRAMES_DIR, output_videos_dir=OUTPUT_VIDEOS_DIR):    
    with open(output_file) as f:
        video_data = json.load(f)

    video_fps = dict()
    food_data = []
    for video in video_data:
        curr = []
        for frames in video['frames']:
            curr.append(frames['questions'][food_item_prompt_index]['answer'])
        curr = [parse_comma_list(answer) for answer in curr]
        video_fps[video['video_name']] = video['fps']
        food_data.append(curr)

    intervals = []
    for video_idx in range(len(food_data)):
        video_name = video_data[video_idx]['video_name']
        curr_video_intervals = create_intervals_optimized(food_data[video_idx], video_name, video_fps)
        intervals.append(curr_video_intervals)
    
    merged_intervals = []
    for i, video_intervals in enumerate(intervals):
        merged_interval = merge_intervals(video_intervals)
        merged_intervals.append(merged_interval)

    pprint.pprint(merged_intervals)

    if not os.path.exists(output_videos_dir):
        os.makedirs(output_videos_dir) 

    for i in range(len(merged_intervals)):
        video_number = i+1
        video_name = f"video_{video_number}.mp4"

        frames_dir = VLM_FRAMES_DIR / video_name
        if include_bboxes:
            frames_dir = POSE_ANNOTATED_FRAMES_DIR / video_name

        output_dir = output_frames_dir / video_name
        output_video_path = output_videos_dir / video_name

        annotate_and_save_frames(frames_dir, output_dir, video_data, video_fps, merged_intervals[i])
        create_video_from_frames(output_dir, output_video_path, video_fps[video_name])




if __name__ == "__main__":
    # generate_vlm_insights()
    # generate_pose_annotations()
    generate_intervals()