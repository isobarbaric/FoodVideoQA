import cv2
from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LLM_DATA_DIR = DATA_DIR / "llm"
LLM_VIDEO_DIR = LLM_DATA_DIR / "videos"
LLM_FRAME_DIR = LLM_DATA_DIR / "frames"

def extract_frames(video_path: Path, 
                   frame_dir: Path, 
                   k: int = 10):
    if not video_path.exists():
        raise ValueError(f"Provided file path {video_path} does not exist")

    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    # load image
    video = cv2.VideoCapture(str(video_path))
    current_frame = 1

    # TODO: raise a relevant error here
    if not video.isOpened():
        pass

    while video.isOpened():
        # videoture each frame
        ret, frame = video.read()

        if ret:
            if current_frame % k == 0:
                name = 'frame' + str(current_frame) + '.jpg'
                # print(f'Creating: {name}')

                # save frame as a jpg file
                cv2.imwrite(str(frame_dir / name), frame)

            # keep track of how many images you end up with
            current_frame += 1
        else:
            break

    video.release()


if __name__ == "__main__":
    video_name = "0.mp4"
    video_path = LLM_VIDEO_DIR / video_name
    frame_path = LLM_FRAME_DIR / video_name

    extract_frames(video_path, frame_path)