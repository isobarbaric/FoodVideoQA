import cv2
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip
import os
from pathlib import Path
import json
from vlm.intervals.parser import parse_comma_list
from vlm.intervals.frame_diff import create_intervals_optimized, merge_intervals
from hyperparameters import FRAME_STEP_SIZE, FOOD_ITEM_IDX
import pprint

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'data_final'
VLM_DATA_DIR = DATA_DIR / 'vlm'
VLM_FRAME_DIR = VLM_DATA_DIR / 'frames'

POSE_DATA_DIR = DATA_DIR / 'pose'
POSE_FRAME_DIR = POSE_DATA_DIR  / 'annotated_frames'

INTERVALS_DATA_DIR = DATA_DIR / 'intervals'
FRAME_OUTPUT_DIR = INTERVALS_DATA_DIR / 'output_frames'
VIDEO_OUTPUT_DIR = INTERVALS_DATA_DIR / 'output_videos'

def overlay_text_on_frame(frame, text, position=(10, 50), scale=1, color=(0, 255, 0), thickness=2):
    font = cv2.FONT_HERSHEY_COMPLEX 
    cv2.putText(frame, text, position, font, scale, color, thickness, lineType=cv2.LINE_AA)
    return frame

def annotate_and_save_frames(frames_dir, output_dir, video_data, video_fps, intervals):
    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
                         key=lambda x: int(x[5:-4]))

    video_name = Path(frames_dir).name
    frame_rate = video_fps[video_name]
    video_json = next(item for item in video_data if item['video_name'] == video_name)

    current_interval_idx = 0

    for frame_data in video_json['frames']:
        frame_number = frame_data['frame_number']
        frame_file = f'frame{frame_number}.jpg'

        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Skipping invalid frame {frame_file}")
            continue

        if frame_data.get("eating", 0) == 1:
            if current_interval_idx < len(intervals):
                start, end, label = intervals[current_interval_idx]
                if start <= frame_number / frame_rate <= end:
                    eating_label = f"Eating: {label}"
                elif frame_number / frame_rate > end:
                    current_interval_idx += 1
                    if current_interval_idx < len(intervals) and label:
                        eating_label = f"Eating: {label}"
                    else:
                        eating_label = "Eating: Unknown"
            else:
                eating_label = "Eating: Unknown"
        else:
            eating_label = "Not Eating"

        print(eating_label)

        frame = overlay_text_on_frame(frame, eating_label, position=(10, 50))

        output_frame_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(output_frame_path, frame)


def create_video_from_frames(frames_dir, output_video_path, original_fps):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
                         key=lambda x: int(x[5:-4]))
    
    frames = [os.path.join(frames_dir, f) for f in frame_files]
    clip = ImageSequenceClip(frames, fps=original_fps / FRAME_STEP_SIZE)
    clip.write_videofile(str(output_video_path), codec='libx264')

if __name__ == "__main__":
    with open(DATA_DIR / 'data.json') as f:
        video_data = json.load(f)

    video_fps = dict()
    food_data = []
    for video in video_data:
        curr = []
        for frames in video['frames']:
            curr.append(frames['questions'][FOOD_ITEM_IDX]['answer'])
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

    for i in range(len(merged_intervals)):
        video_number = i+1
        video_name = f"video_{video_number}.mp4"
        frames_dir = POSE_FRAME_DIR / video_name
        output_dir = FRAME_OUTPUT_DIR / video_name
        output_video_path = VIDEO_OUTPUT_DIR / video_name

        annotate_and_save_frames(frames_dir, output_dir, video_data, video_fps, merged_intervals[i])
        create_video_from_frames(output_dir, output_video_path, video_fps[video_name])
