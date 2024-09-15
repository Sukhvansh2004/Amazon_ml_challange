from transformers import Blip2Processor, Blip2Model
import os
from PIL import Image
import torch
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the BLIP-2 model and processor
model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b").to(device="cuda")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Function to load images from a directory (unchanged)
def load_images_from_directory(directory):
    image_list = []
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # You can add more extensions if needed
            img_path = os.path.join(directory, filename)
            image = Image.open(img_path).convert("RGB")
            image_list.append(image)
            image_paths.append(img_path)
    return image_list, image_paths

# Directory containing the images
image_directory = "images"

# Load and preprocess the images
images, image_paths = load_images_from_directory(image_directory)
inputs = processor(images=images, return_tensors="pt", padding=True).to(device="cuda")

# Get image encodings from BLIP-2 model
with torch.no_grad():
    image_encodings = model.get_image_features(**inputs)

# Convert image encodings to CPU for clustering
image_encodings = image_encodings.cpu().numpy()

# Normalize the encodings for clustering
image_encodings = image_encodings / np.linalg.norm(image_encodings, axis=1, keepdims=True)

# Perform clustering (e.g., using DBSCAN)
clustering_model = DBSCAN(eps=0.1, min_samples=2, metric='cosine')  # Adjust eps as needed
labels = clustering_model.fit_predict(image_encodings)

# Group images into sets based on clusters
image_groups = {}
for label, img_path in zip(labels, image_paths):
    if label not in image_groups:
        image_groups[label] = []
    image_groups[label].append(img_path)

# Save each group of images into separate CSV or text files
output_directory = "clustered_images"
os.makedirs(output_directory, exist_ok=True)

for label, group in image_groups.items():
    if label != -1:  # -1 indicates noise in DBSCAN (unclustered)
        output_file = os.path.join(output_directory, f"cluster_{label}.csv")
        pd.DataFrame(group, columns=["Image Path"]).to_csv(output_file, index=False)
    else:
        # Save the noise group separately if needed
        output_file = os.path.join(output_directory, "noise_group.csv")
        pd.DataFrame(group, columns=["Image Path"]).to_csv(output_file, index=False)

# Optional: Visualize the clusters in 2D using PCA
pca = PCA(n_components=2)
reduced_encodings = pca.fit_transform(image_encodings)

plt.figure(figsize=(10, 7))
plt.scatter(reduced_encodings[:, 0], reduced_encodings[:, 1], c=labels, cmap='tab10', s=100)
plt.colorbar()
plt.title("Image Clusters Visualization")
plt.savefig(os.path.join(output_directory, "Graph.png"))  # Save the visualization graph
