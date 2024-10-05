from transformers import LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM
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
  """
  Load and initialize the specified model and processor from the Hugging Face library.

  Args:
      model_name (str): The name of the model to load.

  Returns:
      tuple: A tuple containing:
          - processor (PreTrainedProcessor): The processor for the specified model.
          - model (PreTrainedModel): The model instance for the specified model.
          - device (torch.device): The device to which the model is loaded (CPU or CUDA).
  
  Raises:
      ValueError: If the provided model_name is not supported.
  """
  if model_name not in SUPPORTED_MODELS:
    raise ValueError(f"{model_name} model not supported; supported models are {SUPPORTED_MODELS}")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  match model_name:
    case "liuhaotian/llava-v1.5-7b":
      processor = AutoProcessor.from_pretrained(model_name)
      model = AutoModelForCausalLM.from_pretrained(model_name)
    case "llava-hf/llava-1.5-7b-hf":
      processor = AutoProcessor.from_pretrained(model_name)
      model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    case "llava-hf/llava-v1.6-mistral-7b-hf":
      processor = LlavaNextProcessor.from_pretrained(model_name)
      model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16) 
    case _:
      processor = AutoProcessor.from_pretrained(model_name)
      model = AutoModel(model_name)
    model.to(device)
    
  # disabling status message doing the same implicitly
  model.generation_config.pad_token_id = model.generation_config.eos_token_id

  return processor, model, device


def make_get_response(model_name: str = DEFAULT_MODEL):
  """
  Create a function to generate responses from a model given a prompt and an image.

  Args:
      model_name (str, optional): The name of the model to use. Defaults to DEFAULT_MODEL.

  Returns:
      function: A function that takes a prompt and an image path, and returns the model's response.
  """
  processor, model, device = get_model(model_name)


  def clean_response(response: str):
    """
    Clean and format the model's response by extracting the relevant answer from the response text.

    Args:
        response (str): The raw response text from the model.

    Returns:
        str: The cleaned and formatted response.
    """
    answer = response.split('ASSISTANT:')[-1]
    return answer.strip()
  

  def get_response(prompt: str, img_path: Path, max_new_tokens: int = 75):
    """
    Generate a response from the model based on a prompt and an image.

    Args:
        prompt (str): The prompt to be processed by the model.
        img_path (Path): The path to the image file.
        max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 75.

    Returns:
        str: The model's response to the prompt based on the image.
    
    Raises:
        ValueError: If the image path does not exist.
    """
    if not img_path.exists():
      raise ValueError(f"Image path {img_path} does not exist")

    image = Image.open(img_path)
    model_prompt = f"USER: <image> {prompt}\nASSISTANT:"
    inputs = processor(text=model_prompt, images=[image])
    inputs.to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
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