from transformers import LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
from pathlib import Path
from typing import Literal, get_args
from rich.console import Console
import time

models = Literal["liuhaotian/llava-v1.5-7b", "llava-hf/llava-1.5-7b-hf", "llava-hf/llava-v1.6-mistral-7b-hf"]
SUPPORTED_MODELS = get_args(models)

DEFAULT_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"


def get_model(model_name: str):
  if model_name not in SUPPORTED_MODELS:
    raise ValueError(f"{model_name} model not supported; supported models are {SUPPORTED_MODELS}")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  match model_name:
    case "liuhaotian/llava-v1.5-7b":
      processor = AutoProcessor.from_pretrained(model_name)
      model = AutoModelForCausalLM.from_pretrained(model_name)
      model.to(device)
    case "llava-hf/llava-1.5-7b-hf":
      processor = AutoProcessor.from_pretrained(model_name)
      model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
      model.to(device)
    case "llava-hf/llava-v1.6-mistral-7b-hf":
      processor = LlavaNextProcessor.from_pretrained(model_name)
      model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16) 
      model.to(device)
    case _:
      raise ValueError(f"Model {model_name} not configured yet")
    
  # disabling status message doing the same implicitly
  model.generation_config.pad_token_id = model.generation_config.eos_token_id

  return processor, model, device


# TODO: turn prompt into prompts
def make_get_response(model_name: str = DEFAULT_MODEL):
  processor, model, device = get_model(model_name)

  def clean_response(response: str):
    answer = response.split('ASSISTANT:')[-1]
    return answer.strip()

  def get_response(prompt: str, img_path: Path, max_new_tokens: int = 75):
    if not img_path.exists():
      raise ValueError(f"Image path {img_path} does not exist")

    image = Image.open(img_path)
    model_prompt = f"USER: <image> {prompt}\nASSISTANT:"
    inputs = processor(text=model_prompt, images=[image])
    inputs.to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # special tokens are tokens that the model adds to your response to generate a response
    output = processor.decode(output[0], skip_special_tokens=True)

    return clean_response(output)
  
  return get_response


if __name__ == "__main__":
  console = Console()

  prompt = "What is shown in this image?"
  img_path = Path("eat.jpg")
  
  model_name = "llava-hf/llava-1.5-7b-hf"
  get_response = make_get_response(model_name)

  start = time.time()
  response = get_response(prompt, img_path)
  end = time.time()

  console.print(f"[green]{response}[/green]\n")
  print(f"get_response: {end - start} seconds\n")