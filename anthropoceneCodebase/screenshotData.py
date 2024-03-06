import requests
from PIL import Image
import os
from transformers.models.blip import BlipProcessor, BlipForConditionalGeneration


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

img_url = '/projects/p32234/projects/aerith/anthropocene_ML/aws_try_2/2023_backup/projects/multimedia_anthropocene/object_detection/screenshot_data/0A0BC23A/Trashed00001.jpeg' 
raw_image = Image.open(img_url).convert('RGB')

# conditional image captioning
text = "anthropocene"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

img_dir = '/projects/p32234/projects/aerith/anthropocene_ML/aws_try_2/2023_backup/projects/multimedia_anthropocene/object_detection/screenshot_data/0A0BC23A/'
for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img)
    raw_image = Image.open(img_path).convert('RGB')
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_length=1000)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(caption)
    with open('/projects/p32234/projects/aerith/anthropocene_ML/results/image_captioning_extending_output_length.csv', 'a') as f:
        f.write(img + ', ' + str(caption).replace(",", "") + '\n')



