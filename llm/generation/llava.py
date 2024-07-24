from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
from pathlib import Path

# using LLaVA-v1.6
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.generation_config.pad_token_id = model.generation_config.eos_token_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# TODO: turn prompt into prompts
# TODO: get Cache error to go away
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
  inputs = processor(rev_prompt, image, return_tensors="pt").to("cuda")

  # the .generate() function is the origin of the `past_key_values` bug
  output = model.generate(**inputs, max_new_tokens=100)
  model_response = processor.decode(output[0], skip_special_tokens=True) 

  actual_response = model_response.split('[/INST]')[1].strip()
  return actual_response


if __name__ == "__main__":
  prompt = "What is shown in this image?"
  img_path = Path("eat.jpg")
  
  response = get_response(prompt, img_path)
  print(response)