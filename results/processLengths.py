import pandas as pd
import re

# Load the CSV file
file_path = 'results_height.csv'  # Update with the actual file path
df = pd.read_csv(file_path)

# Function to process the prediction and apply transformations
def process_prediction(prediction):
    # Pattern to match decimal or whole numbers followed by a unit of length (with plural cases and variations)
    pattern = re.compile(r'(\d+\.?\d*)\s*(cm|centimeters?|inches?|in|ft|feet|foot|m|metres?|mm|millimetres?|yd|yards?)', re.IGNORECASE)
    
    # Search for a match in the prediction text
    match = pattern.search(prediction)
    
    if match:
        # Extract the value and unit from the match
        value, unit = match.groups()
        
        # Normalize the unit to singular long form
        unit_map = {
            'cm': 'centimetre',
            'centimeter': 'centimetre',
            'centimeters': 'centimetre',
            'in': 'inch',
            'inch': 'inch',
            'inches': 'inch',
            'ft': 'foot',
            'feet': 'foot',
            'foot': 'foot',
            'm': 'metre',
            'metre': 'metre',
            'metres': 'metre',
            'mm': 'millimetre',
            'millimetre': 'millimetre',
            'millimetres': 'millimetre',
            'yd': 'yard',
            'yard': 'yard',
            'yards': 'yard'
        }
        
        # Convert to the correct unit (singular form)
        unit = unit_map[unit.lower()]
        
        # Return the normalized value and unit
        return f"{value} {unit}"
    else:
        # If no match is found or the sentence is not valid, return an empty string
        return ""

# Apply the function to the 'predicted_value' column
df['predicted_value'] = df['predicted_value'].apply(process_prediction)

# Save the processed file
output_file_path = 'cleaned_results_file_height.csv'  # Update the file path to save the new CSV
df.to_csv(output_file_path, index=False)

print(f"File saved at: {output_file_path}")
