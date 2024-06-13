from pytube import YouTube
import fire
from pathlib import Path
from rich.console import Console


def download_video(youtube_url: str, output_path: str = 'video.mp4'):
    console = Console()

    youtube = YouTube(youtube_url)
    output_path = Path(output_path)

    filename = output_path.name
    output_path = output_path.parent

    video = youtube.streams.get_highest_resolution()
    video.download(output_path=output_path, filename=filename)

    console.print(f"[yellow]YouTube video {youtube_url} downloaded successfully[/yellow]")


if __name__ == "__main__":
    fire.Fire(download_video)