import cv2
from pathlib import Path
import shutil
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LLM_DATA_DIR = DATA_DIR / "llm"
LLM_VIDEO_DIR = LLM_DATA_DIR / "videos"
LLM_FRAME_DIR = LLM_DATA_DIR / "frames"

def extract_frames(video_path: Path, 
                   frame_dir: Path = LLM_FRAME_DIR, 
                   k: int = 10):
    """
    Extract frames from a video file and save every k-th frame as a JPEG image.

    Args:
        video_path (Path): The path to the video file from which frames will be extracted.
        frame_dir (Path): The directory where the extracted frames will be saved.
        k (int, optional): The interval at which frames will be saved (e.g., every k-th frame). Defaults to 10.

    Raises:
        ValueError: If the provided video file path does not exist.
        RuntimeError: If the video file cannot be opened by OpenCV.
    """
    if not video_path.exists():
        raise ValueError(f"Provided file path {video_path} does not exist")

    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    # load image
    video = cv2.VideoCapture(str(video_path))
    current_frame = 1

    if not video.isOpened():
        raise RuntimeError(f"Cannot open video file {video_path}")

    while video.isOpened():
        # videoture each frame
        ret, frame = video.read()

        if ret:
            if current_frame % k == 0:
                name = 'frame' + str(current_frame) + '.jpg'
                # save frame as a jpg file
                cv2.imwrite(str(frame_dir / name), frame)

            # keep track of how many images you end up with
            current_frame += 1
        else:
            break

    video.release()


def extract_random_frames(video_path: Path, 
                   frame_dir: Path = LLM_FRAME_DIR, 
                   num_frames: int = 10):
    """
    Extract random frames from a video file and save them as JPEG images.
    """

    print(f"Extracting {num_frames} random frames from video: {video_path}")

    if not video_path.exists():
        raise ValueError(f"Provided file path {video_path} does not exist")

    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(1, num_frames+1):
        frame_number = np.random.randint(0, total_frames)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()

        if ret:
            name = f"frame_{i}.jpg"
            cv2.imwrite(str(frame_dir / name), frame)
        else:
            break

    video.release()


if __name__ == "__main__":
    video_name = "0.mp4"
    video_path = LLM_VIDEO_DIR / video_name
    frame_path = LLM_FRAME_DIR / video_name

    extract_frames(video_path, frame_path)