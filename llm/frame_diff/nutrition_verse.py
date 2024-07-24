"""
this file uses data from data.json

command to run this code: 'torchrun nutrition_verse.py
"""

from llama import Dialog, Llama
from pathlib import Path
from typing import List, Optional
import json
import pprint
import os

with open("data.json", 'r') as file:
    data = json.load(file)


def main(
    prompts: List[str],
    ckpt_dir: str = "Meta-Llama-3-8B/",
    tokenizer_path: str = "Meta-Llama-3-8B/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4, # this is 6 in the demo code
    max_gen_len: Optional[int] = None,
) -> List[dict]:
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = []
    for prompt in prompts:
        dialogs.append({"role": "user", "content": prompt})

    dialogs = [dialogs]

#     dialogs: List[Dialog] = [
#         [{"role": "user", 
#           "content": "what is the recipe of mayonnaise?"}],
#         [
#             {"role": "user", 
#              "content": "I am going to Paris, what should I see?"},
#             {"role": "assistant",
#                 "content": """\
# Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

# 1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
# 2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
# 3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

# These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
#             },
#             {"role": "user", "content": "What is so great about #1?"},
#         ],
#         [
#             {"role": "system", "content": "Always answer with Haiku"},
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#         ],
#         [
#             {
#                 "role": "system",
#                 "content": "Always answer with emojis",
#             },
#             {"role": "user", "content": "How to go from Beijing to NY?"},
#         ],
#     ]

    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # answer = []
    # for prompt, result in zip(prompts, results):
    #     entry = {}
    #     entry['prompt'] = prompt
    #     entry['output'] = result['generation'].strip()
    #     answer.append(entry)
    
    # return answer
    return results

def get_video_frames(): # video_name: str):
    # video_data = None
    
    # for video in data:
    #     if video_name == video['video_name']:
    #         video_data = video['frames']

    # if not video_data:
    #     raise ValueError(f"No video exists with {video_name}")
    
    frames_dict = {}

    for video in data:
        video_name = video["video_name"]
        frames_dict[video_name] = {}

        for i in range(4):
            frames_dict[video_name][f"question_{i}"] = []

        for frame in video["frames"]:
            for i in range(4):
                frames_dict[video_name][f"question_{i}"].append(frame["questions"][i]["answer"])

    return frames_dict


def get_frame_diff(frame1: str, 
                   frame2: str):
    # prompts = [
    #     f"Describe any changes that have occurred between the following two frames of a video: \n\nFrame #1: {frame1} \n\nFrame #2: {frame2}"
    # ]
    prompts = [
        f"Describe any changes that have occurred between the following two frames of a video. I will give you these 2 frames in the following 2 prompts.",
        f"This is frame #1: {frame1}",
        f"This is frame #2: {frame2}"
    ]

    results = main(prompts)
    return results


if __name__ == "__main__":
    frames = get_video_frames()

    test_frame1 = frames['0.mp4']['question_0'][0]
    test_frame2 = frames['0.mp4']['question_0'][1]

    results = get_frame_diff(test_frame1, test_frame2)
    pprint.pprint(results)
