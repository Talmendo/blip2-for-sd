import requests, torch, sys, os

from importlib import reload
from PIL import Image
from transformers import Blip2Processor, BlipProcessor, Blip2ForConditionalGeneration
from tqdm import tqdm

import caption_processor

def load_model(model_name):
  print(f"Loading Model {model_name}")
  if model_name == "Salesforce/blip2-opt-2.7b":
    print("WARNING: Salesforce/blip2-opt-2.7b will give worse results, consider using Salesforce/blip2-opt-6.7b-coco instead if possible.")
  processor = Blip2Processor.from_pretrained(model_name)
  model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)

  device = None
  if torch.cuda.is_available():
    print("CUDA available, using GPU")
    device = "cuda"
  else:
    print("CUDA not available, using CPU")
    device = "cpu"

  print("Moving model to device")
  model = model.to(device)
  print("Model loaded")

  return (model, processor, device)

class CaptionForSD:
  def __init__(self, model, processor, device, config=None):
    self.update_caption_processor(model, processor, device, config)

  def update_caption_processor(self, model, processor, device, config=None):
    reload(caption_processor)
    self.caption_processor_ref = caption_processor.CaptionProcessor(model, processor, device, config)

  def test(self):
    images = [Image.open("img/01.jpg"), Image.open("img/02.jpg")]

    captions = [
      self.caption_processor_ref.caption_me_formatted("photo of a woman", images[0]),
      self.caption_processor_ref.caption_me_formatted("photo of a woman", images[1]),
    ]

    return (images, captions)

  def run(self, path):
    prompt_file_dict = {}

    # list all sub dirs in path
    sub_dirs = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]

    print("Reading prompts from sub dirs and finding image files")
    for prompt in sub_dirs:
      prompt_file_dict[prompt] = [file for file in os.listdir(os.path.join(path, prompt)) if file.endswith((".jpg", ".png", ".jpeg", ".webp"))]

    for prompt, file_list in prompt_file_dict.items():
      print(f"Found {str(len(file_list))} files for prompt \"{prompt}\"")

    for prompt, file_list in prompt_file_dict.items():
      total = len(file_list)

      for file in tqdm(file_list):
        # read image
        image = Image.open(os.path.join(path, prompt, file))

        caption = ""
        # generate caption
        try:
          caption = self.caption_processor_ref.caption_me_formatted(prompt, image)
        except:
          print("Error creating caption for file: " + file)

        # save caption to file
        # file without extension
        with open(os.path.join(path, prompt, os.path.splitext(file)[0] + ".txt"), "w", encoding="utf-8") as f:
          f.write(caption)

    print("Done")
