import torch, re

class CaptionProcessor:
  def __init__(self, model, processor, device, config=None):
    self.model = model
    self.processor = processor
    self.device = device
    self.config = config
    self.batch = {}
  def gen_from_config_with_override(self, inputs, override_config=None):
    config = self.config

    if override_config is not None:
      for key, value in override_config.items():
        config[key] = value

    return self.gen_from_config(inputs, config)

  def gen_from_config(self, inputs, config):
    return self.gen(
      inputs,
      max_length=config["max_length"],
      min_length=config["min_length"],
      top_k=config["top_k"],
      top_p=config["top_p"],
      num_beams=config["num_beams"],
      repetition_penalty=config["repetition_penalty"],
      no_repeat_ngram_size=config["max_repititions"],
      do_sample=config["do_sample"],
    )

  def gen(self, inputs, max_length=10, min_length=0, top_k=30, top_p=0.92, num_beams=4, repetition_penalty=1.5, no_repeat_ngram_size=2, do_sample=False):
    return self.model.generate(
      **inputs,
      # max_new_tokens=25,                        # Number of tokens to generate
      max_length=max_length,                      # Maximum length of the sequence to be generated, mutually exclusive with max_new_tokens
      num_beams=num_beams,                        # Number of beams to use for beam search
      num_return_sequences=1,                     # Number of captions to generate
      early_stopping=True,                        # Stop when no new tokens are generated
      repetition_penalty=repetition_penalty,      # Penalize repeated words
      no_repeat_ngram_size=no_repeat_ngram_size,  # Number of words that can be repeated
      do_sample=do_sample,                        # Introduce randomness to captions
      # temperature=0.9,                          # Measure of randomness 0-1, 0 means no randomness
      top_k=top_k,                                # Number of highest probability tokens to keep, 0 means no filtering
      top_p=top_p,                                # Probability threshold, 0 means no filtering
      min_length=min_length,                      # Minimum length of the sequence to be generated
    )

  def process(self, prompt, image):
    return self.processor(image, text=prompt, return_tensors="pt", padding=True, truncation=True, max_length=75).to(self.device, torch.bfloat16)

  def caption_from(self, generated):
    caption_list = self.processor.batch_decode(generated, skip_special_tokens=True)
    caption_list = [caption.strip() for caption in caption_list]
    return caption_list if len(caption_list) > 1 else caption_list[0]

  def sanitise_caption(self, caption):
    replace_dict = {
      "wearing nothing": "nude",
      "wearing naked":   "nude",
      "wearing wearing": "wearing",
      ", shot on$":      "",
    }

    caption = caption.split("Answer:")[0].strip().lower()

    for key, value in replace_dict.items():
      caption = re.sub(key, value, caption)

    return caption

  def sanitise_prompt_part(self, prompt):
    replace_dict = {
      r",\s*(and)?":                                  " and ",
      r"a point and shoot[\w\s]*,?":                  "",
      r"(she|he|it|(a|the)?\s*subject) is a?\s*":     "",
      "wearing nothing":                              "nude",
      r"\s(on|in|at|for)\s?a?\s?,":                   ",",
      r"between the choices\s?":                      "",
    }
    
    if not type(prompt) is list:
      prompt = [prompt]

    for i in range(len(prompt)):
      prompt[i] = prompt[i].split("Answer:")[0].strip().lower()

      for key, value in replace_dict.items():
        prompt[i] = re.sub(key, value, prompt[i])
  
    return prompt

  def ask(self, question, image, override_config=None):
    self.batch[image.filename] = f"Question: {question} Answer:"
    processed = self.process(f"Question: {question} Answer:", image)

    generated = None

    if self.config is None:
      generated = self.gen(processed)
    if override_config is not None:
      generated = self.gen_from_config_with_override(processed, override_config)
    else:
      generated = self.gen_from_config(processed, self.config)

    return self.sanitise_prompt_part(self.caption_from(generated))[0]

  def caption_me(self, initial_prompt, image):
    prompt = initial_prompt
    try:
      prompt = self.caption_from(self.gen(self.process(initial_prompt, image), max_length=75, min_length=35, top_k=30, top_p=0.92, num_beams=4))
    except Exception as e:
      print(e)
    return f"{initial_prompt}, {prompt}"

  def caption_me_formatted(self, initial_prompt, image, details=""):
    prompt = ""

    q = {
      "hair_color":   ("What is her hair color?", None),
      "hair_length":  ("What is her hair length?", None),
      "style":        ("Between the choices selfie, mirror selfie, candid, professional portrait what is the style of the photo?", None),
      "clothing":     ("What is the subject wearing if anything?", None),
      "action":       ("What is the subject doing? Be succint", {"max_length": 15}),
      "framing":      ("Between the choices closeup, upper body shot, full body shot what is the framing of the photo?", None),
      "setting":      ("Describe the setting and background of the image, what can you see? Be descriptive and vivid", {"max_length": 25}),
      "lighting":     ("What is the lighting like? Use professional terms, like: soft lighting, studio lighting, natural lighting and so on", None),
      # "angle":        ("What angle is the picture taken from? Be succint, like: from the side, from below, from front", None),
      # "camera":       ("What kind of camera could this picture have been taken with? Be specific and guess a brand with specific camera type", None),
    }

    a = {}

    try:
      for key, question in q.items():
        a[key] = self.ask(question[0], image, override_config=question[1])

      if details:
        details = f"{details}, "

      # prompt = self.sanitise_caption(f"{a['style']}, {initial_prompt}, with {a['hair_color']} {a['hair_length']} hair, wearing {a['clothing']}, {a['action']}, {details} {a['framing']}, {a['setting']}, {a['lighting']}, {a['angle']}, shot on {a['camera']}")
      prompt = self.sanitise_caption(f"{a['style']}, {initial_prompt}, with {a['hair_color']} {a['hair_length']} hair, wearing {a['clothing']}, {a['action']}, {details} {a['framing']}, {a['setting']}, {a['lighting']}")
    except Exception as e:
      print(e)
    
    return prompt