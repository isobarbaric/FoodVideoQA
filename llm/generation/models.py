from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, pipeline
import torch
from PIL import Image
from pathlib import Path
from typing import Literal, get_args
from rich.console import Console
import time

# using LLaVA-v1.6
models = Literal["llava-hf/llava-1.5-7b-hf", "llava-hf/llava-v1.6-mistral-7b-hf"]
SUPPORTED_MODELS = get_args(models)

def get_model(model_name: str):
  # if model_name not in SUPPORTED_MODELS:
  #   raise ValueError(f"{model_name} model not supported; supported models are {SUPPORTED_MODELS}")

  processor, model = None, None
  match model_name:
    case "llava-hf/llava-v1.6-mistral-7b-hf":
      processor = LlavaNextProcessor.from_pretrained(model_name)
      model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16) 
      model.generation_config.pad_token_id = model.generation_config.eos_token_id

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  return processor, model, device


# TODO: turn prompt into prompts
# TODO: get Cache error to go away
def make_get_response(model_name: str):
  processor, model, device = get_model(model_name)

  def get_response(prompt: str, img_path: Path):
    image = Image.open(img_path)
    # each value in "content" has to be a list of dicts with types ("text", "image")
    conversation = [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": prompt},
              {"type": "image"},
            ],
        }
    ]
    rev_prompt = processor.apply_chat_template(conversation)

    # pt = return PyTorch tensor
    inputs = processor(rev_prompt, image, return_tensors="pt")
    inputs.to(device)
    print(f"inputs: {inputs.keys()}")

    model.to(device)
    # the .generate() function is the origin of the `past_key_values` bug
    output = model.generate(**inputs, max_new_tokens=100)
    model_response = processor.decode(output[0], skip_special_tokens=True) 

    actual_response = model_response.split('[/INST]')[1].strip()
    return actual_response
  
  return get_response


def test(model_name, img_path, prompt):
  # model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  pipe = pipeline("image-to-text", model=model_name, device=device)

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

  processor = AutoProcessor.from_pretrained(model_name)
  prompt = processor.apply_chat_template(conversation)
  outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

  response = outputs[0]['generated_text']
  actual_response = response.split('[/INST]')[1].strip()
  return actual_response


if __name__ == "__main__":
  console = Console()
  prompt = "What is shown in this image?"
  img_path = Path("eat.jpg")

  model_name = "llava-hf/llava-v1.6-mistral-7b-hf" 
  get_response = make_get_response(model_name)

  start = time.time()
  response = get_response(prompt, img_path)
  end = time.time()
  print(f"[green]get_response: {response}[/green]\n")
  print(f"get_response: {end - start} seconds\n")

  start = time.time()
  response = test(model_name, img_path, prompt)
  end = time.time()
  print(f"[green]test: {response}[/green]\n")
  print(f"test: {end - start} seconds\n")