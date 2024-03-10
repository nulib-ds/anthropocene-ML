import os
from PIL import Image
from transformers.models.blip import BlipProcessor, BlipForConditionalGeneration


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

base_dir = '/projects/p32234/projects/aerith/anthropocene_ML/aws_try_2/2023_backup/projects/multimedia_anthropocene/object_detection/screenshot_data/'
text = ""
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_length=1000)
        caption = processor.decode(out[0], skip_special_tokens=True)
        print(caption)
        with open('/projects/p32234/projects/aerith/anthropocene_ML/results/image_captioning_screenshot_data.csv', 'a') as f:
            f.write(str(image_path), ',"' + str(caption), '"', '\n')
