from pytubefix import YouTube
import fire
from rich.console import Console
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LLM_DATA_DIR = DATA_DIR / "llm"
LLM_VIDEO_DIR = LLM_DATA_DIR / "videos"

console = Console()
    
def download_video(youtube_url: str, output_path: Path = LLM_VIDEO_DIR / "video_1.mp4"):
    output_path = Path(output_path)

    if output_path.exists():
        raise ValueError(f"A video already exists at the provided output path {output_path}")

    try:
        youtube = YouTube(youtube_url)
        output_path = Path(output_path)

        filename = output_path.name
        output_path = output_path.parent

        video = youtube.streams.get_highest_resolution()
        video.download(output_path=output_path, filename=filename)

        console.print(f"[yellow]YouTube video {youtube_url} downloaded successfully[/yellow]")
    except Exception as e:
        console.print(f"[red]An error occurred: {e}[/red]")


def trim_video(video_path: str, start_time: int, end_time: int):
    video_path = Path(video_path)
    temp_output_path = video_path.parent / "temp.mp4"

    ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=temp_output_path)

    video_path.unlink()
    temp_output_path.rename(video_path)

    console.print(f"[yellow]Trimmed video {video_path} successfully; video starts at {start_time} and ends at {end_time}[/yellow]")


# TODO: Redo the CLI
if __name__ == "__main__":
    fire.Fire({
        '--download': download_video,
        '--trim': trim_video,
    })