from gpt4all import GPT4All
import json
from rich.console import Console
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LLM_DIR = DATA_DIR / "llm"

INPUT_PATH = LLM_DIR / "data.json"
OUTPUT_PATH = LLM_DIR / "frame_diff.json"

MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

def load_frames():
    with open(INPUT_PATH, 'r') as file:
        data = json.load(file)

    frames_dict = {}

    for video in data:
        video_name = video["video_name"]
        frames_dict[video_name] = []

        for frame in video["frames"]:
            frame_info = {"frame_number": frame["frame_number"], "questions": []}
            for i in range(len(frame["questions"])):
                frame_info["questions"].append(frame["questions"][i]["answer"])
            frames_dict[video_name].append(frame_info)

    return frames_dict

"""
Convert response into a dictionary:
Example response:
{
    "new": ["fried foods", "fried chicken", "chicken wings/drumsticks"],
    "absent": []
}
"""
def convert_to_dict(response):
    index_open = response.find("{")
    index_close = response.rfind("}") + 1
    response = response[index_open:index_close]

    try:
        response_dict = json.loads(response)

    except json.JSONDecodeError:
        response_dict = {
            "new": [],
            "absent": []
        }

    return response_dict

def generate_frame_diff():
    console = Console()
    model = GPT4All(MODEL_NAME)
    frames_dict = load_frames()

    output_dict = {}

    with model.chat_session():
        frame_diff_prompt =   """
                    You are analyzing changes between two consecutive frames in a video. Focus ONLY on the presence of NEW food items and the absence of OLD food items between the two descriptions.

                    Please provide your answer in the following structured distionary format:

                    {
                        "new": [List of food items that appear in the current frame but were not present in the previous frame],
                        "absent": [List of food items that were present in the previous frame but are missing in the current frame]
                    }

                    Only mention the food items that are NEW or ABSENT. If there are none in either list, leave the list empty. Do NOT include any other information in your response.
                    """

        for video_name, frames in frames_dict.items():
            output_dict[video_name] = []
            for i in range(1, len(frames)):
                prev_frame = frames[i - 1]
                curr_frame = frames[i]

                prev_desc = "\n\n Previous Description: " + prev_frame["questions"][0]
                curr_desc = "\n\n Current Description: " + curr_frame["questions"][0]

                prompt = frame_diff_prompt + prev_desc + curr_desc
                console.print(f"[yellow]PROMPT:[/yellow]")
                console.print(f"[green]{prompt}[/green]")
                print()

                response = model.generate(prompt=prompt, temp=0)
                console.print(f"[yellow]RESPONSE:[/yellow]")
                console.print(f"[blue]{response}[/blue]\n\n")
                print()

                response_dict = convert_to_dict(response)

                output_dict[video_name].append({
                    "frame_number": curr_frame["frame_number"],
                    "difference": response_dict
                })

    with open(OUTPUT_PATH, 'w') as outfile:
        json.dump(output_dict, outfile, indent=4)


def determine_eaten():
    console = Console()

    with open(OUTPUT_PATH, 'r') as file:
        data = json.load(file)

    absent = []
    for video_name, frames in data.items():
        console.print(f"[yellow]Absent food items in {video_name}[/yellow]\n")
        for frame in frames:
            diff = frame["difference"]
            if diff["absent"]:
                console.print(f"[red]Frame Number: {frame['frame_number']}[/red]")
                for item in diff["absent"]:
                    print(f"- {item}")
                    absent.append(item)
                print()


if __name__ == "__main__":
    start = time.time()
    generate_frame_diff()
    end = time.time()
    print(f"{end - start} seconds elapsed...")
    determine_eaten()