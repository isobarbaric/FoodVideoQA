import cv2
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip
import os
from pathlib import Path
import json
from vlm.intervals.parser import parse_comma_list
from hyperparameters import FRAME_STEP_SIZE
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
LLM_DATA_DIR = DATA_DIR / 'llm'
LLM_FRAME_DIR = LLM_DATA_DIR / 'frames'
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

def overlay_text_on_frame(frame, text, position=(10, 50), scale=1, color=(0, 255, 0), thickness=2):
    font = cv2.FONT_HERSHEY_COMPLEX 
    cv2.putText(frame, text, position, font, scale, color, thickness, lineType=cv2.LINE_AA)
    return frame

def annotate_and_save_frames(frames_dir, intervals, output_dir, frame_rate):
    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
                         key=lambda x: int(x[5:-4]))
    current_interval_idx = 0

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame_num = int(frame_file[5:-4])

        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Skipping invalid frame {frame_file}")
            continue

        if current_interval_idx < len(intervals):
            start, end, label = intervals[current_interval_idx]
            if start <= frame_num / frame_rate <= end:
                frame = overlay_text_on_frame(frame, label)
                print("overlayed: ", label)
            elif frame_num / frame_rate > end:
                current_interval_idx += 1
                if current_interval_idx < len(intervals):
                    start, end, label = intervals[current_interval_idx]
                    if start <= frame_num / frame_rate <= end:
                        frame = overlay_text_on_frame(frame, label)

        output_frame_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(output_frame_path, frame)

def create_video_from_frames(frames_dir, output_video_path, original_fps):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
                         key=lambda x: int(x[5:-4]))
    
    frames = [os.path.join(frames_dir, f) for f in frame_files]
    clip = ImageSequenceClip(frames, fps=original_fps/FRAME_STEP_SIZE)
    clip.write_videofile(output_video_path, codec='libx264')

frames_dir = LLM_FRAME_DIR / 'video_2.mp4'
output_dir = 'stitcher_output_frames'
frame_rate = video_fps['video_2.mp4']
intervals = [
    [0.0, 2.0, 'soup'],
    [4.67, 5.33, 'water'],
    [8.0, 9.33, 'seed'],
    [9.33, 12.67, 'pancake'],
    [13.33, 15.33, 'spinach'],
    [15.33, 20.0, 'pizza'],
    [20.0, 20.67, 'onion'],
    [20.67, 23.33, 'pizza'],
    [24.0, 24.67, 'carrot'],
    [24.67, 25.33, 'shrimp'],
    [26.67, 27.33, 'carrot'],
    [30.0, 31.33, 'soup'],
    [31.33, 32.0, 'coffee'],
    [32.0, 32.67, 'cake'],
    [34.0, 34.67, 'soup'],
    [35.33, 38.0, 'cake'],
    [38.67, 40.0, 'cheese']
]
annotate_and_save_frames(frames_dir, intervals, output_dir, frame_rate)

output_video_path = 'output_video_with_labels.mp4'
create_video_from_frames(output_dir, output_video_path, frame_rate)
