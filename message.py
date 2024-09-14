import pandas as pd
import numpy as np
import os

# Load the dataset
df = pd.read_csv(r'dataset/train.csv')

# Create directories for storing files
import os
if not os.path.exists('image_links'):
    os.makedirs('image_links')

# Extract unique entity names
unique_entities = df['entity_name'].unique()

for entity in unique_entities:
    # Filter by entity_name and sample 100 image links
    filtered_df = df[df['entity_name'] == entity]
    sampled_df = filtered_df.sample(n=100, random_state=42)  # Sample 100 random rows
    
    # Save image links and entity values
    sampled_df[['image_link']].to_csv(f'image_links/{entity}_image_links.csv', index=False)
    sampled_df[['entity_value', 'image_link']].to_csv(f'image_links/{entity}_entity_value_links.csv', index=False)



from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
import pandas as pd
import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
# Define your model and configuration with the system prompt
model = 'OpenGVLab/InternVL2-4B'
offload_folder = 'model_offload'
os.makedirs(offload_folder, exist_ok=True)

# Update the configuration with the offload folder
config = TurbomindEngineConfig(
    session_len=8192,
    temperature=0.7,
    system_prompt=(
        "You are a multimodal chatbot capable of analyzing both visual and textual content from images. "
        "Your task is to extract relevant information by:\n\n"
        "Image Text Analysis: Use OCR to extract any text present within the image.\n"
        "Visual Cue Analysis: Identify any additional visual cues such as logos, colors, objects, or shapes that could assist in answering the query.\n\n"
        "After processing both the text and the visual elements:\n\n"
        "If the extracted data provides enough context to respond meaningfully, answer the query with the values with no explanation of your reasoning.\n"
        "If no relevant data is found or the query cannot be answered based on the image content, simply return value 0."
    ),
    offload_folder=offload_folder  # Specify the offload folder
)

try:
    pipe = pipeline(model, backend_config=config)
except Exception as e:
    print(f"An error occurred while creating the pipeline: {e}")
    raise

# Function to get predictions
def get_prediction(image_link, entity_name):
    image = load_image(image_link)
    query = f"What is the {entity_name} of the product?"
    response = pipe((query, image))
    
    # Extract only the value from the response
    return response.text.strip()

# Process each entity_name
unique_entities = ['weight', 'wattage', 'depth', 'height', 'width', 'voltage', 'maximum_weight_recommendation', 'volume']  # Replace with actual unique entity names

for entity in unique_entities:
    # Load the CSV files
    df = pd.read_csv(f'image_links/{entity}_entity_value_links.csv')
    
    # Prepare lists to store results
    results = []
    
    for _, row in df.iterrows():
        image_link = row['image_link']
        actual_value = row['entity_value']
        predicted_value = get_prediction(image_link, entity)
        
        # Store result
        results.append({
            'image_link': image_link,
            'actual_value': actual_value,
            'predicted_value': predicted_value
        })
    
    # Save results to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(f'image_links/{entity}_results.csv', index=False)