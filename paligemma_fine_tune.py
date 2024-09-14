import os
import random
import pandas as pd
from src.constants import allowed_units 
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import requests
from peft import LoraConfig, get_peft_model, TaskType

# Set random seed for reproducibility
random.seed(42)

model_id = "google/paligemma-3b-mix-224"

# Load the model and processor
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval().to(device="cuda")
processor = AutoProcessor.from_pretrained(model_id)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # For sequence-to-sequence tasks
    inference_mode=False,
    r=8,  # Rank of the LoRA matrix
    lora_alpha=32,
    lora_dropout=0.1
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)

def predictor(image_link, category_id, entity_name, entity_value):
    image = Image.open(requests.get(image_link, stream=True).raw)
    prompt = f"What are the {entity_name} of this product? You are allowed to use the below given units only and if the f{entity_name} of the product isn't specified output a blank string \n {allowed_units}"
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device='cuda')
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        print(decoded, '\n', f"Ground Truth is {entity_value}")
        return decoded

def train_model(train_df, epochs=3, lr=1e-4):
    # Prepare optimizer and training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for _, row in train_df.iterrows():
            # Get the image and prompt
            image = Image.open(requests.get(row['image_link'], stream=True).raw)
            prompt = f"What are the {row['entity_name']} of this product?"
            
            # Prepare inputs for model
            model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device='cuda')
            labels = processor(text=row['entity_value'], return_tensors="pt").input_ids.to(device='cuda')

            # Forward pass
            outputs = model(**model_inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_df)}")

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset'
    
    # Load the dataset
    train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    
    # Specify the number of data points to randomly sample for training
    N = 1000  # Change N to the desired number of samples

    # Randomly sample N rows from the dataframe for training
    train_sample = train.sample(n=N, random_state=42)

    # Fine-tune the model with LoRA
    train_model(train_sample, epochs=3, lr=1e-4)
    
    # Inference on sampled test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Apply the predictor function to the sampled rows
    test_sample = test.sample(n=N, random_state=42)
    test_sample['prediction'] = test_sample.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name'], row['entity_value']), axis=1)
    
    # Save the output
    output_filename = os.path.join(DATASET_FOLDER, 'sample_paligemma_test_out.csv')
    test_sample[['index', 'prediction']].to_csv(output_filename, index=False)
