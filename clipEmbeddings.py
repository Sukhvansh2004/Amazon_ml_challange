from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch 

# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device="cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and preprocess the image
image1 = Image.open("new_image.jpg")
image2 = Image.open("car.jpg")
inputs = processor(images=[image1, image2], return_tensors="pt").to(device="cuda")

# Get image encodings
with torch.no_grad():
    image_encodings = model.get_image_features(**inputs)

print(image_encodings)
