import cv2
from pathlib import Path

def extract_frames(video_path: Path, output_dir: Path, k: int = 5):
    if not video_path.exists():
        raise ValueError(f"No video exists at the provided file path {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

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
                print(f'Creating: {name}')

                # save frame as a jpg file
                cv2.imwrite(str(extracted_path / name), frame)

            # keep track of how many images you end up with
            current_frame += 1
        else:
            break

    video.release()


if __name__ == "__main__":
    video_name = "0.mp4"
    video_path = Path("custom-videos") / video_name

    extracted_path = Path("extracted-frames") / video_name
    extracted_path.mkdir(parents=True, exist_ok=True)

    extract_frames(video_path, extracted_path)