from pytube import YouTube
import fire
from pathlib import Path
from rich.console import Console
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip 

console = Console()

def download_video(youtube_url: str, output_path: str):
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


# download_video seems to only download the video, so remove_audio isn't necessary for the timebeing 
# def remove_audio(video_path: Path):
#     video = VideoFileClip(video_path)
#     video_without_audio = video.without_audio()
#     video_without_audio.write_videofile(video_path, codec="libx264")


if __name__ == "__main__":
    fire.Fire({
        '--download': download_video,
        '--trim': trim_video,
        # 'remove-audio': remove_audio
    })