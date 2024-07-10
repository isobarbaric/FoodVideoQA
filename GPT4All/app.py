from gpt4all import GPT4All
import json
from rich.console import Console
import pprint
import time

"""
idk why i did this but

frames_dict = {"*.mp4" : {
                            "question_0": [...responses...], 
                            "question_1": [...responses...],
                            "question_2": [...responses...],
                            "question_3": [...responses...]
                        }
                }
where ...responses... are across all frames
"""

def load_frames():
    with open("data.json", 'r') as file:
        data = json.load(file)

    frames_dict = {}

    for video in data:
        video_name = video["video_name"]
        frames_dict[video_name] = {}

        for i in range(4):
            frames_dict[video_name][f"question_{i}"] = []

        for frame in video["frames"]:
            for i in range(4):
                frames_dict[video_name][f"question_{i}"].append(frame["questions"][i]["answer"])
            
        # remove later
        break

    return frames_dict


if __name__ == "__main__":
    console = Console()
    
    start = time.time()
    model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
    frames_dict = load_frames()
    with model.chat_session():
        general = "Given the following descriptions, find what has CHANGED between the current description and the previous one: "

        prompts = frames_dict["0.mp4"]["question_0"]

        for i in range(1, len(prompts)-1):
            prev = "\n\n Previous Description: " +  prompts[i-1]
            curr = "\n\n Current Description: " + prompts[i]

            prompt = general + prev + curr
            console.print(f"[orange]PROMPT:[/orange]")
            console.print(f"[green]{prompt}[/green]")

            response = model.generate(prompt = prompt, temp = 0)
            console.print(f"[orange]RESPONSE:[/orange]")
            console.print(f"[blue]{response}[/blue]\n\n")

    end = time.time()
    print(f"{end - start} seconds elapsed...")