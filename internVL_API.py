import os
import csv
import requests
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from internvl.models import InternVLModel
from urllib.parse import urlparse
import pandas as pd

# Directory to save downloaded images
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load the InternVL2-8B model
model = InternVLModel(model_name="internvl2-8b")

# Function to download an image from a URL
def download_image(image_url):
    try:
        img_name = os.path.basename(urlparse(image_url).path)
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        if not os.path.exists(img_path):  # Skip download if image already exists
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                print(f"Image downloaded: {img_path}")
            else:
                print(f"Failed to download {image_url}")
                return None
        return img_path
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")
        return None

# Function to process an image with the model
def process_image(image_path, prompt):
    try:
        if image_path:
            print(f"Processing image: {image_path}")
            
            # Process the image with InternVL2-8B model
            result = model.predict(image_path, prompt)  # Assuming the model has a predict method with prompt support
            print(f"Prediction for {image_path}: {result}")
            return result
        else:
            print(f"Skipping image {image_path} due to download failure.")
            return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to process each row from the CSV file
def process_row(index, image_url, group_id, entity_name):
    image_path = download_image(image_url)
    prompt = f"Provide details about the {entity_name}"  # Custom prompt based on entity_name
    result = process_image(image_path, prompt)
    
    return {
        "index": index,
        "group_id": group_id,
        "entity_name": entity_name,
        "prediction": result
    }

# Function to process all rows from the CSV file
def process_csv(input_csv, output_csv):
    results = []
    
    # Read the CSV file
    with open(input_csv, mode='r') as file:
        reader = csv.DictReader(file)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    process_row, row['index'], row['image_link'], row['group_id'], row['entity_name']
                ) for row in reader
            ]
            for future in futures:
                try:
                    result = future.result()  # This will raise any exceptions that occurred during execution
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error in future: {e}")
    
    # Save the results to a new CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Main execution function
if __name__ == "__main__":
    input_csv = r"dataset\sample_test_mini.csv"  # Replace with the path to your input CSV file
    output_csv = r"dataset\output_intervlAPI.csv"  # Output CSV file to store predictions
    
    try:
        process_csv(input_csv, output_csv)
    except Exception as e:
        print(f"Error in processing images: {e}")
