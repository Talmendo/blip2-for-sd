from importlib import reload

import caption_for_sd
import caption_processor

model = None
processor = None
device = None

config = {
  "max_length": 15,             # not the total max length, but the max length per prompt part
  "min_length": 0,
  "top_k": 30,
  "top_p": 0.92,
  "do_sample": False,
  "num_beams": 4,
  "repetition_penalty": 1.5,
  "max_repititions": 2
}

def load_model(model_name):
  global model, processor, device

  model, processor, device = caption_for_sd.load_model(model_name)
  
def main(path):
  # reloading caption_processor to enable us to change its values in between executions
  # without having to reload the model, which can take very long
  reload(caption_processor)
  caption_for_sd.CaptionForSD(model, processor, device, config).run(path)

if __name__ == "__main__":
  # strongly recommend to use "Salesforce/blip2-opt-6.7b-coco" for better results, Salesforce/blip2-opt-2.7b will be much worse
  load_model("Salesforce/blip2-opt-6.7b-coco")

  while True:
    print("Enter path: ")
    path = input()
    main(path)