import os
import random
import pandas as pd
from src.constants import entity_unit_map 
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import requests

# Set random seed for reproducibility
random.seed(42)

model_id = "google/paligemma-3b-ft-vqav2-448"

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval().to(device="cuda")
processor = AutoProcessor.from_pretrained(model_id)

def predictor(image_link, category_id, entity_name, entity_value):
    image = Image.open(requests.get(image_link, stream=True).raw)
    prompt = f"What is the {entity_name} of this product? You are allowed to use the below given units only and if the {entity_name} of the product isn't specified output a blank string \n {entity_unit_map}.\n Do not use short forms."
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device='cuda')
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        print(f"The image is {image_link} and attribute is {entity_name} \n",decoded, '\n', f"Ground Truth is {entity_value}")
        return decoded

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset'
    
    # Load the dataset
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    
    # Specify the number of data points to randomly sample
    N = 1000  # Change N to the desired number of samples

    # Randomly sample N rows from the dataframe
    test_sample = test.sample(n=N, random_state=42)

    # Apply the predictor function to the sampled rows
    test_sample['prediction'] = test_sample.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name'], row['entity_value']), axis=1)
    
    # Save the output
    output_filename = os.path.join(DATASET_FOLDER, 'sample_paligemma_test_out.csv')
    test_sample[['index', 'image_link', 'entity_name', 'prediction']].to_csv(output_filename, index=False)
