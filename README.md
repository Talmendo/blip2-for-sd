# blip2-for-sd

Simple script to make BLIP2 output image description in a format suitable for Stable Diffusion.

Format followed is roughly
`[STYLE OF PHOTO] photo of a [SUBJECT], [IMPORTANT FEATURE], [MORE DETAILS], [POSE OR ACTION], [FRAMING], [SETTING/BACKGROUND], [LIGHTING], [CAMERA ANGLE], [CAMERA PROPERTIES],in style of [PHOTOGRAPHER]`

## Usage

Install dependencies according to requirements.txt

**Recommended**: Use the Jupyter Notebook `caption.ipynb` 

**Alternatively**

- run main.py
  `python main.py`

The default model will be loaded automatically from huggingface.
You will be presented with an input to specify the folder to process after the model is loaded.

<img width="854" alt="Screenshot 2023-08-04 102650" src="https://github.com/Talmendo/blip2-for-sd/assets/141401796/fa40cae5-90a4-4dd5-be1d-fc0e8312251a">

**Folder Structure**: The image or source folder should have the following structure:

![Screenshot 2023-08-04 102544](https://github.com/Talmendo/blip2-for-sd/assets/141401796/eea9c2b0-e96a-40e4-8a6d-32dd7aa3e802)

Each folder represents a base prompt to be used for every image inside.

**Custom Keywords**
Add a `image_name.details.txt` file next to an image to inject specific keywords to the prompt. This is useful if you want to do some manual captioning but don't want to bother describing the whole image or if you need to make sure certain keywords are present.
Example:
image001.png
image001.details.txt

## Models

Default model is  `Salesforce/blip2-opt-6.7b-coco`. Requires ~20GB of VRAM unless run with bitsandbytes, then  only ~8GB. See below.
Also tested with `Salesforce/blip2-opt-2.7b` which seems to give much worse results, but is also less demanding on your hardware and a bit faster.

## Saving VRAM

4bit Quantization is supported. Simply set use_4bit=True when calling load_model
You need bitsandbytes for that. Probably won't work on Windows.
