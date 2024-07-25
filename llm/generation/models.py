from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
from pathlib import Path
from typing import Literal, get_args

# using LLaVA-v1.6
models = Literal["llava-hf/llava-v1.6-mistral-7b-hf"]
SUPPORTED_MODELS = get_args(models)

def get_model(model_name: str):
  if model_name not in SUPPORTED_MODELS:
    raise ValueError(f"{model_name} model not supported; supported models are {SUPPORTED_MODELS}")

  processor, model = None, None
  match model_name:
    case "llava-hf/llava-v1.6-mistral-7b-hf":
      processor = LlavaNextProcessor.from_pretrained(model_name)
      model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16) 

  model.generation_config.pad_token_id = model.generation_config.eos_token_id
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  return processor, model


# TODO: turn prompt into prompts
# TODO: get Cache error to go away
def make_get_response(model_name: str):
  processor, model = get_model(model_name)

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
    rev_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    print(f"with generation prompt: {processor.apply_chat_template(conversation, add_generation_prompt=True)}")
    print(f"with generation prompt: {processor.apply_chat_template(conversation)}")

    inputs = processor(rev_prompt, image, return_tensors="pt").to("cuda")

    # the .generate() function is the origin of the `past_key_values` bug
    output = model.generate(**inputs, max_new_tokens=100)
    model_response = processor.decode(output[0], skip_special_tokens=True) 

    actual_response = model_response.split('[/INST]')[1].strip()
    return actual_response
  
  return get_response


if __name__ == "__main__":
  prompt = "What is shown in this image?"
  img_path = Path("eat.jpg")

  model_name = "llava-hf/llava-v1.6-mistral-7b-hf" 
  get_response = make_get_response(model_name)

  response = get_response(prompt, img_path)
  print(response)