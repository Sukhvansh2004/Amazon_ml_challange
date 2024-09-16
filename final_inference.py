import pandas as pd
import os
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

path = "testImages"
# Function to get predictions
def get_prediction(image_name, entity_name, pipe):
    #image = load_image(image_name)
    modified_file_path = image_name.split('/')[-1]
    image = load_image(f'{path}/{modified_file_path}')
    query = f"""What is the {entity_name} of the product? Only return the numerical value with the unit. Follow the following return format:
    Output only the extracted value in the format "x unit," where x is a float and unit is from the allowed list below.
    - Do not include any additional text except for units.
    Example: Return "12.0 inch" or "500 gram" (without additional phrases or short forms).
    """
    response = pipe((query, image))
    return response.text.strip()

# Process each entity-specific CSV file
entity_files = [r'maximum_weight_recommendation.csv']
batch_size = 100  # Save results after processing every 100 rows
i = 1

for entity_file in entity_files:
    # Extract entity_name from the filename
    entity_name = os.path.splitext(entity_file)[0]
    
    # Reload the model and configuration for each entity_name
    model = 'OpenGVLab/InternVL2-4B'
    offload_folder = 'model_offload'
    os.makedirs(offload_folder, exist_ok=True)

    config = TurbomindEngineConfig(
        session_len=8192,
        temperature=0.7,
        system_prompt=(
            """Task: Extract relevant information from the image based on the following:

    1. Data Extraction: Extract width, height, depth, item_weight, item_volume, wattage, voltage, or maximum_weight_recommendation using OCR and visual cues.

    2. Return Format:
    - Output only the extracted value in the format "x unit," where x is a float and unit is from the allowed list below.
    - Do not include any additional text like "The prediction is."
    - Example: Return "12.0 inch" or "500 gram" (without additional phrases or short forms).
    - Do not use commas for large numbers. For example, return "10000.0" instead of "10,000."

    3. Allowed Units:
    {
        'width': ['centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'],
        'depth': ['centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'],
        'height': ['centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'],
        'voltage': ['kilovolt', 'millivolt', 'volt'],
        'wattage': ['kilowatt', 'watt'],
        'item_volume': ['centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                        'imperial gallon', 'litre', 'millilitre', 'pint', 'quart'],
        'weight' : ['gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'],
        'maximum_weight_recommendation' : ['gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton']
    }

    4. Examples:
    - Image shows "Height: 12 inches" → Return: "12.0 inch"
    - g to gram, W to Watt
    - No relevant data → Return: " "
    """),
        offload_folder=offload_folder
    )

    # Load the model and pipeline
    try:
        pipe = pipeline(model, backend_config=config)
    except Exception as e:
        print(f"An error occurred while creating the pipeline: {e}")
        continue

    # Load the CSV file for the current entity_name
    entity_df = pd.read_csv(os.path.join('entity_csvs', entity_file), index_col=0)  # Use the original index

    # Check if results file already exists to resume
    output_filename = f'results_{entity_name}.csv'
    if os.path.exists(output_filename):
        processed_df = pd.read_csv(output_filename)
        # Get the index of the last processed row
        last_processed_index = processed_df['original_index'].max()
        # Filter out already processed rows
        entity_df = entity_df[entity_df.index > last_processed_index]
    else:
        processed_df = pd.DataFrame()

    # Prepare lists to store results
    results = []

    # Iterate over each row in the filtered dataframe
    for _, row in entity_df.iterrows():
        original_index = row.name  # Original index from the DataFrame
        image_link = row['image_link']
        
        # Get prediction for the current image and entity
        predicted_value = get_prediction(image_link, entity_name, pipe)
        
        # Store result with original row index and predicted value
        results.append({
            'original_index': original_index,
            'image_link': image_link,
            'predicted_value': predicted_value
        })
        
        i += 1
        # Save results after processing every 'batch_size' rows or at the end of the loop
        if len(results) >= batch_size:
            # Append new results to the CSV file
            results_df = pd.DataFrame(results)
            processed_df = pd.concat([processed_df, results_df], ignore_index=True)
            processed_df.to_csv(output_filename, index=False)
            print(f"Saved {i} results to {output_filename}")
            results = []  # Clear the results list

#        print(f"Processed row {original_index} for entity '{entity_name}': Predicted Value: {predicted_value}")

    # Save any remaining results at the end of processing
    if results:
        results_df = pd.DataFrame(results)
        processed_df = pd.concat([processed_df, results_df], ignore_index=True)
        processed_df.to_csv(output_filename, index=False)
        print(f"Final results for entity '{entity_name}' saved to {output_filename}")
