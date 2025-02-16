from rich.console import Console
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pathlib import Path
import click
import yt_dlp

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LLM_DATA_DIR = DATA_DIR / "llm"
LLM_VIDEO_DIR = LLM_DATA_DIR / "videos"


def download_video(
    youtube_url: str,
    output_path: Path = LLM_VIDEO_DIR / "video.mp4",
    exists_ok: bool = False,
):
    """Download a video from YouTube.

    This command downloads a video from the provided YouTube URL and saves it to the specified output path.

    Args:
        youtube_url (str):
            The URL of the YouTube video you want to download. This argument is required.
    Options:
        --output-path (str):
            The file path where the downloaded video will be saved.
    """
    output_path = str(output_path)

    if Path(output_path).exists():
        if not exists_ok:
            raise ValueError(
                f"A video already exists at the provided output path {output_path}"
            )
        else:
            Path(output_path).unlink()

    try:
        print(f"Downloading video from YouTube URL: {youtube_url}")
        ydl_opts = {
            "outtmpl": output_path,
            "format": "best",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        print(f"Downloaded video successfully to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def trim_video(video_path: str, start_time: int, end_time: int):
    """Trim a video file to a specified duration.

    This command allows you to trim a video file from a start time to an end time. The output will be a trimmed version of the original video file.

    Args:
        video_path (str):
            Path to the video file that you want to trim. The file must exist.
        start_time (int):
            The start time in seconds from which the video should be trimmed. The value should be a non-negative integer and should be less than the end time.
        end_time (int):
            The end time in seconds at which the video should stop. The value should be greater than the start time and should be within the duration of the video.
    """
    video_path = Path(video_path)
    temp_output_path = video_path.parent / "temp.mp4"

    ffmpeg_extract_subclip(
        video_path, start_time, end_time, targetname=temp_output_path
    )

    video_path.unlink()
    temp_output_path.rename(video_path)

    click.echo(
        f"[yellow]Trimmed video {video_path} successfully; video starts at {start_time} and ends at {end_time}[/yellow]"
    )


@click.group()
def cli():
    pass


@cli.command()
@click.argument("--youtube-url", required=True, type=str)
@click.option(
    "--output-path",
    type=click.Path(exists=True),
    help="Path to save the downloaded video.",
)
def download(youtube_url: str, output_path: Path = None):
    download_video(youtube_url, output_path)


@cli.command()
@click.argument("video_path", required=True, type=click.Path(exists=True))
@click.argument("start_time", required=True, type=int)
@click.argument("end_time", required=True, type=int)
def trim(video_path: str, start_time: int, end_time: int):
    trim_video(video_path, start_time, end_time)


if __name__ == "__main__":
    cli()
