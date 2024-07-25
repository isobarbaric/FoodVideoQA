from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, pipeline
import torch
from PIL import Image
from pathlib import Path
from typing import Literal, get_args
from rich.console import Console
import time

models = Literal["llava-hf/llava-1.5-7b-hf", "llava-hf/llava-v1.6-mistral-7b-hf"]
SUPPORTED_MODELS = get_args(models)


def get_model(model_name: str):
  if model_name not in SUPPORTED_MODELS:
    raise ValueError(f"{model_name} model not supported; supported models are {SUPPORTED_MODELS}")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  pipe = pipeline("image-to-text", model=model_name, device=device)
  processor = AutoProcessor.from_pretrained(model_name)

  return processor, pipe


# TODO: turn prompt into prompts
# TODO: get Cache error to go away
def make_get_response(model_name: str):
  processor, pipe = get_model(model_name)

  def get_response(prompt: str, img_path: Path, max_new_tokens: int = 75):
    # each value in "content" has to be a list of dicts with types ("text", "image")
    image = Image.open(img_path)
    conversation = [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": prompt},
              {"type": "image"},
            ],
        }
    ]

    prompt = processor.apply_chat_template(conversation)
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})

    response = outputs[0]['generated_text']
    actual_response = response.split('[/INST]')[1].strip()
    return actual_response
  
  return get_response


if __name__ == "__main__":
  console = Console()

  prompt = "What is shown in this image?"
  img_path = Path("eat.jpg")
  model_name = "llava-hf/llava-v1.6-mistral-7b-hf" 

  get_response = make_get_response(model_name)

  start = time.time()
  response = get_response(prompt, img_path)
  end = time.time()

  console.print(f"[green]{response}[/green]\n")
  print(f"get_response: {end - start} seconds\n")