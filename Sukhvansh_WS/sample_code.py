import os
import random
import pandas as pd
from src.constants import allowed_units 
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import requests

model_id = "google/paligemma-3b-mix-224"

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval().to(device="cuda")
processor = AutoProcessor.from_pretrained(model_id)

def predictor(image_link, category_id, entity_name):
    image = Image.open(requests.get(image_link, stream=True).raw)
    prompt = f"What are the {entity_name} of this product? You are allowed to use the below given units only and if the f{entity_name} of the product is'nt specified output a blank string \n {allowed_units}"
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device='cuda')
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset'
    
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test_mini.csv'))
    
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'sample_paligemma_test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)  