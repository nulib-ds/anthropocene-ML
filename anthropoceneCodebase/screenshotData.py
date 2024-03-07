import os
import pandas as pd
from PIL import Image
from transformers.models.blip import BlipProcessor, BlipForConditionalGeneration


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")



output_dataframe = pd.DataFrame(columns=['image_id', 'caption', 'film_id'])

img_dir = '/projects/p32234/projects/aerith/anthropocene_ML/aws_try_2/2023_backup/projects/multimedia_anthropocene/object_detection/screenshot_data/'
text = ""
for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img)
    raw_image = Image.open(img_path).convert('RGB')
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_length=1000)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(caption)
    with open('/projects/p32234/projects/aerith/anthropocene_ML/results/image_captioning_extending_output_length.csv', 'a') as f:
        f.write(img + ', ' + str(caption).replace(",", "") + '\n')



