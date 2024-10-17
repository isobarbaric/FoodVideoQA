import json
from rich.console import Console
import time
from pathlib import Path
from typing import Literal, get_args
import torch
from transformers import LlavaForConditionalGeneration
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
from transformers import ByT5Tokenizer, BartForConditionalGeneration, BartTokenizer, BartModel
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

from pprint import pprint

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data'
LLM_DATA_DIR = DATA_DIR / 'llm'
DATA_JSON = LLM_DATA_DIR / 'data.json'

# all LLMs similar supported by huggingface
models = Literal[
    # very bad, terrible output
    'google/flan-t5-small',
    'google/flan-t5-base',
    'google/flan-t5-large',
    'google/flan-t5-xl',
    'google/flan-t5-xxl',
    'google/flan-ul2',

    'google/byt5-small',
    'google/byt5-base',
    'google/byt5-large',
    'google/byt5-xl',
    'google/byt5-xxl',

    'google-t5/t5-small',
    'google-t5/t5-base',
    'google-t5/t5-large',
    'google-t5/t5-3b',
    'google-t5/t5-11b',

    # bart - bad, hallucinates
    'facebook/bart-base',
    'facebook/bart-large',

    # gpts - not great
    'EleutherAI/gpt-neo-1.3B',
    'EleutherAI/gpt-neo-2.7B',
    'EleutherAI/gpt-j-6B',

    # default - llama3 (great results)
    'meta-llama/Meta-Llama-3-8B-Instruct',

    # amazing (but very very large) - llama3.1 fine-tuned by NVIDIA
    'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
    
    # TODO: experiment with these
    'facebook/opt-350m',
    'bigscience/bloom',
    'Rostlab/prot_t5_xl_uniref50',
    'facebook/m2m100_418M',
    'CohereForAI/aya-101',
]
SUPPORTED_MODELS = get_args(models)

DEFAULT_MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'

def get_model(model_name: str):
    """
    Load and initialize the specified model and tokenizer from the Hugging Face library.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing:
            - tokenizer (PreTrainedTokenizer): The tokenizer for the specified model.
            - model (PreTrainedModel): The model instance for the specified model.
            - device (torch.device): The device to which the model is loaded (CPU or CUDA).
    
    Raises:
        ValueError: If the provided model_name is not supported.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"{model_name} model not supported; supported models are {SUPPORTED_MODELS}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name in ['google/flan-t5-small', 'google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'google/flan-ul2',
                      'google/byt5-small', 'google/byt5-base', 'google/byt5-large', 'google/byt5-xl', 'google/byt5-xxl',
                      'google-t5/t5-small', 'google-t5/t5-base', 'google-t5/t5-large', 'google-t5/t5-3b', 'google-t5/t5-11b']:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif model_name == 'CohereForAI/aya-101':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif model_name == 'Rostlab/prot_t5_xl_uniref50':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_name == 'facebook/m2m100_418M':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_name == 'meta-llama/Meta-Llama-3-8B-Instruct':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_name == 'facebook/bart-base' or model_name == 'facebook/bart-large':
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
    elif model_name == 'EleutherAI/gpt-neo-1.3B' or model_name == 'EleutherAI/gpt-neo-2.7B' or model_name == 'EleutherAI/gpt-j-6B':
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, ignore_mismatched_sizes=True)
        model = GPTNeoForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
    elif model_name == 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    else:
        model = DEFAULT_MODEL
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = LlavaForConditionalGeneration.from_pretrained(model)
    model.to(device)

    model.config.pad_token_id = model.config.eos_token_id

    return tokenizer, model, device



def load_frames(data_path: Path):
    """
    Load and parse frames from a JSON file into a dictionary.

    Returns:
        dict: A dictionary where the keys are video names and the values are lists of frames. Each frame is a dictionary containing frame number and associated questions.
    """
    with open(data_path, 'r') as file:
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

def convert_to_dict(response: str):
    """
    Convert a string response into a structured dictionary with 'new' and 'absent' food items.

    Args:
        response (str): The raw response string to be converted.

    Returns:
        dict: A dictionary with two keys, 'new' and 'absent', each containing a list of food items.

    Example response:
    {
        "new": ["fried foods", "fried chicken", "chicken wings/drumsticks"],
        "absent": ["beans", "rice", "salad"]
    }
    """
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

def load_data(data_path: Path):
    """
    Load and parse data from a JSON file.

    Returns:
        list: A list of dictionaries containing data for each video.
    """
    with open(data_path, 'r') as file:
        data = json.load(file)

    return data

def generate_frame_diff(input_path: Path, output_path: Path, model_name: str = DEFAULT_MODEL, print_output: bool = False):
    """
    Generate a structured dictionary of differences between consecutive video frames, focusing on new and absent food items.

    This function reads frame data from a JSON file, uses a specified model to analyze changes between frames, and saves the results to an output JSON file.
    """
    console = Console()
    tokenizer, model, device = get_model(model_name)

    data_json = load_data(input_path)
    frames_dict = load_frames(input_path)
    output_dict = {}

    def clean_response(response: str):
        answer = response.split('ASSISTANT:')[-1]
        return answer.strip()

    frame_diff_prompt =   """
                You are analyzing changes between two consecutive frames in a video. Focus ONLY on the presence of NEW food items and the absence of OLD food items between the two descriptions.

                Please provide your answer in the following structured dictionary format. ONLY include edible food items in your response.
                {
                    "new": [List of food items that appear in the current frame but were not present in the previous frame. INCLUDE ONLY EDIBLE PREPARED FOOD ITEMS.],
                    "absent": [List of food items that were present in the previous frame but are missing in the current frame. INCLUDE ONLY EDIBLE PREPARED FOOD ITEMS.]
                }

                Only mention the food items that are NEW or ABSENT. If there are none in either list, leave the list empty. Do NOT include any other information in your response.
                """
    
    for video in data_json:
        video_name = video["video_name"]
        frames = frames_dict[video_name]

        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            prev_desc = "\n\n Previous Description: " + prev_frame["questions"][0]
            curr_desc = "\n\n Current Description: " + curr_frame["questions"][0]
            prompt = frame_diff_prompt + prev_desc + curr_desc

            model_prompt = f"USER: {prompt}\nASSISTANT:"
            inputs = tokenizer(text=model_prompt, return_tensors="pt")
            inputs.to(device)
            output = model.generate(**inputs, max_new_tokens=256)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            response = clean_response(response)
            
            if print_output:
                console.print(f"[yellow]PROMPT:[/yellow]")
                console.print(f"[green]{prompt}[/green]")
                print()

                console.print(f"[yellow]RESPONSE:[/yellow]")
                console.print(f"[blue]{response}[/blue]\n\n")
                print()

            response_dict = convert_to_dict(response)

            # add to current video in data.json, and overwrite the file
            video["frames"][i]["difference"] = response_dict

            if video_name not in output_dict:
                output_dict[video_name] = []
            output_dict[video_name].append({"frame_number": curr_frame["frame_number"], "difference": response_dict})
    
    # sort each video's frames by frame number
    for video in output_dict:
        output_dict[video] = sorted(output_dict[video], key=lambda x: x["frame_number"])

    with open(output_path, 'w') as file:
        json.dump(data_json, file, indent=4)
        



def determine_eaten(data_json: Path, print_output: bool = False):
    """
    Analyze the output JSON file to determine which food items are missing in each frame compared to previous frames.

    This function reads the generated frame differences from a JSON file and prints out the absent food items for each frame.
    """
    console = Console()

    with open(data_json, 'r') as file:
        data = json.load(file)

    for video in data:
        if print_output:
            console.print(f"[yellow]Video Name: {video['video_name']}[/yellow]")
            console.print()

        absent = []
        for frame in video["frames"]:
            frame_number = frame["frame_number"]

            if "difference" not in frame:
                frame["difference"] = {
                    "new": [],
                    "absent": []
                }

            difference = frame["difference"]

            if difference["absent"]:

                if print_output:
                    console.print(f"[blue]Frame Number: {frame_number}[/blue]")
                    console.print(f"[red]Absent Food Items:[/red]")
                    for item in difference["absent"]:
                        console.print(f"  - {item}")
                    console.print()
                
                absent.extend(difference["absent"])
        
        video["eaten foods: "] = absent

    with open(data_json, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    start = time.time()
    generate_frame_diff(input_path=DATA_JSON, output_path=LLM_DATA_DIR / 'output.json', print_output=True)
    end = time.time()
    print(f"{end - start} seconds elapsed...")
    determine_eaten()