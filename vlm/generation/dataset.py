from pathlib import Path

from .youtube import download_video, trim_video
from .video_utils import extract_random_frames, extract_frames
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import concatenate_videoclips, VideoFileClip

ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
DATASET_VIDEO_DIR = DATASET_DIR / "videos"
DATASET_FRAME_DIR = DATASET_DIR / "frames"

# taken from 'Video Metadata' Spreadsheet
YOUTUBE_URLS_INTERVALS = {
    "https://www.youtube.com/shorts/jIUr9BGzfhw": "0:14-0:40",
    "https://www.youtube.com/shorts/bjbKLdOiHlI": "",
    "https://www.youtube.com/shorts/MB3_Wv664Xg": "0:05-0:12",
    "https://www.youtube.com/shorts/B5D6glUD-zI": "0:23-0:58",
    "https://www.youtube.com/watch?v=tVb8tO0wbXM": "",
    "https://www.youtube.com/watch?v=muZXCCyBt8A": "0:22-2:35",
    "https://www.youtube.com/watch?v=tdLMrhDk_7k": "",
    "https://www.youtube.com/watch?v=3d1zkBdsc6E": "0:33-2:24",
    "https://www.youtube.com/watch?v=aQ_a1qRVpDs": "0:49-2:49",
    "https://www.youtube.com/watch?v=i1yqpHMaOnw": "0:30-2:15",
    "https://www.youtube.com/watch?v=KZCUpJtwWTE": "0:25-3:20",
    "https://www.youtube.com/watch?v=95NTGBplNKU": "0:26-2:54",
    "https://www.youtube.com/watch?v=-F3rNdcjbbI": "1:35-3:40",
    "https://www.youtube.com/watch?v=f2oNzIe8IN4": "0:30-2:35",
    "https://www.youtube.com/watch?v=bvxIC3fgKRs": "1:00-3:30",
    "https://www.youtube.com/watch?v=aetBmlakJO8": "0:50-2:25",
    "https://www.youtube.com/watch?v=dUEMXUMwiAw": "3:57-5:40, 11:54-13:30, 18:00-22:40",
    "https://www.youtube.com/watch?v=3Nof3K6771k": "0:33-2:53",
    "https://www.youtube.com/watch?v=B0hpWKANgpk": "0:40-3:30",
    "https://www.youtube.com/watch?v=x1EhKV6e0-8": "0:48-1:33, 3:33-4:18",
    "https://www.youtube.com/watch?v=2wJz3MSE9Jw": "0:00-2:54",
    "https://www.youtube.com/watch?v=IYWkulMVAdU": "0:32-2:03, 2:15-2:25",
    "https://www.youtube.com/watch?v=CkgiFn4--0k": "0:17-2:40",
    "https://www.youtube.com/watch?v=3CSgxAE9Htg": "1:25-5:10",
    "https://www.youtube.com/watch?v=Mmmmhnh64gA": "1:45-4:15"
}


def download_all():
    """Download all videos from the specified YouTube URLs."""
    for i, youtube_url in enumerate(YOUTUBE_URLS_INTERVALS, start=1):
        output_path = DATASET_VIDEO_DIR / f"video_{i}.mp4"
        download_video(youtube_url, output_path, exists_ok=True)

def trim_video(video_path: str, start_time: int, end_time: int):
    """Trim a video file to a specified duration."""
    temp_output_path = video_path.parent / "temp.mp4"
    ffmpeg_extract_subclip(str(video_path), start_time, end_time, targetname=str(temp_output_path))
    return temp_output_path

def trim_all():
    for i, (youtube_url, intervals) in enumerate(YOUTUBE_URLS_INTERVALS.items(), start=1):
        video_path = DATASET_VIDEO_DIR / f"video_{i}.mp4"
        intervals_list = intervals.split(", ")

        clips = []
        for interval in intervals_list:
            if interval:
                start_str, end_str = interval.split("-")
                start_time = int(start_str.split(":")[0]) * 60 + int(start_str.split(":")[1])
                end_time = int(end_str.split(":")[0]) * 60 + int(end_str.split(":")[1])

                trimmed_clip_path = trim_video(video_path, start_time, end_time)
                clip = VideoFileClip(str(trimmed_clip_path))
                clips.append(clip)

        if clips:
            final_clip = concatenate_videoclips(clips)
            final_clip_output_path = DATASET_VIDEO_DIR / f"video_{i}.mp4"
            final_clip.write_videofile(str(final_clip_output_path), codec="libx264")

            for clip in clips:
                clip.close()

            trimmed_clip_path.unlink(missing_ok=True)


def slice_random_frames(num_frames: int = 20):
    """Slice random frames from all the downloaded videos."""
    for i in range(1, len(YOUTUBE_URLS_INTERVALS) + 1):
        video_path = DATASET_VIDEO_DIR / f"video_{i}.mp4"
        frame_dir = DATASET_FRAME_DIR / f"video_{i}"
        extract_random_frames(video_path, frame_dir, num_frames=num_frames)

def slice_consecutive_frames(frame_interval: int = 20):
    """Slice frames from all the downloaded videos."""
    for i in range(1, len(YOUTUBE_URLS_INTERVALS) + 1):
        video_path = DATASET_VIDEO_DIR / f"video_{i}.mp4"
        frame_dir = DATASET_FRAME_DIR / f"video_{i}"
        extract_frames(video_path, frame_dir, k=frame_interval)

if __name__ == "__main__":
    # download_all()
    # trim_all()
    slice_consecutive_frames()