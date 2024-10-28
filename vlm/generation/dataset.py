from pathlib import Path

from .youtube import download_video, trim_video
from .video_utils import extract_random_frames

ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
DATASET_VIDEO_DIR = DATASET_DIR / "videos"
DATASET_FRAME_DIR = DATASET_DIR / "frames"

# taken from 'Video Metadata' Spreadsheet
YOUTUBE_URLS = [
    "https://www.youtube.com/shorts/jIUr9BGzfhw",
    "https://www.youtube.com/shorts/bjbKLdOiHlI",
    "https://www.youtube.com/shorts/MB3_Wv664Xg",
    "https://www.youtube.com/shorts/iUa2kZfYij8",
    "https://www.youtube.com/shorts/B5D6glUD-zI",
    "https://www.youtube.com/watch?v=3CjI6-frbIw",
    "https://www.youtube.com/watch?v=tVb8tO0wbXM",
    "https://www.youtube.com/watch?v=muZXCCyBt8A",
    "https://www.youtube.com/watch?v=tdLMrhDk_7k",
    "https://www.youtube.com/watch?v=PA3ljNl8GOs",
    "https://www.youtube.com/watch?v=3d1zkBdsc6E",
    "https://www.youtube.com/watch?v=aQ_a1qRVpDs",
    "https://www.youtube.com/watch?v=i1yqpHMaOnw",
    "https://www.youtube.com/watch?v=KZCUpJtwWTE",
    "https://www.youtube.com/watch?v=aXBrutEnieE",
    "https://www.youtube.com/watch?v=95NTGBplNKU",
    "https://www.youtube.com/watch?v=wJsrpd8N9EM",
    "https://www.youtube.com/watch?v=-F3rNdcjbbI",
    "https://www.youtube.com/watch?v=2Bcv_1HiZpQ",
    "https://www.youtube.com/watch?v=f2oNzIe8IN4",
    "https://www.youtube.com/watch?v=AGzLYuXW5u8",
    "https://www.youtube.com/watch?v=bvxIC3fgKRs",
    "https://www.youtube.com/watch?v=aetBmlakJO8",
    "https://www.youtube.com/watch?v=I3gIKW_SsRQ",
    "https://www.youtube.com/watch?v=dUEMXUMwiAw",
    "https://www.youtube.com/watch?v=muypwhe6ICw",
    "https://www.youtube.com/watch?v=3Nof3K6771k",
    "https://www.youtube.com/watch?v=3jxwnVI2A2g",
    "https://www.youtube.com/watch?v=B0hpWKANgpk",
    "https://www.youtube.com/watch?v=x1EhKV6e0-8",
    "https://www.youtube.com/watch?v=Qpv8wR-Pzbc",
    "https://www.youtube.com/watch?v=U458ae4qAW4",
    "https://www.youtube.com/watch?v=_qmnPZL5Mw4",
    "https://www.youtube.com/watch?v=FrQ5R7HAMIY",
    "https://www.youtube.com/watch?v=r9-nA0DNvqg",
]


def download_all_videos():
    """Download all the YouTube videos in the YOUTUBE_URLS list."""
    for i, youtube_url in enumerate(YOUTUBE_URLS, start=1):
        output_path = DATASET_VIDEO_DIR / f"video_{i}.mp4"
        download_video(youtube_url, output_path, exists_ok=True)

def slice_random_frames():
    """Slice random frames from all the downloaded videos."""
    for i in range(1, len(YOUTUBE_URLS) + 1):
        video_path = DATASET_VIDEO_DIR / f"video_{i}.mp4"
        frame_dir = DATASET_FRAME_DIR / f"video_{i}"
        extract_random_frames(video_path, frame_dir, num_frames=10)

if __name__ == "__main__":
    # download_all_videos()
    slice_random_frames()