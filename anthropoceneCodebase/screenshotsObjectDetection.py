import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection
import os

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Directory containing the JPEG images
directory = "/home/ysc4337/aerith/anthropocene_ML/aws_try_2/2023_backup/projects/multimedia_anthropocene/object_detection/screenshot_data/"

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Construct the full path to the image file
        filepath = os.path.join(directory, filename)

        # Open the image file
        image = Image.open(filepath)

        # Perform object detection on the image
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        # Print detected objects and rescaled box coordinates
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
