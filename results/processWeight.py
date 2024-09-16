import pandas as pd
import re

# Load the CSV file
file_path = 'results_maximum_weight_recommendation.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Function to extract and clean the first valid weight value with full unit expansion
def clean_weight_value(value):
    if pd.isna(value):
        return ""  # Return empty string for NaN values
    
    # Regular expressions for weight units and their full forms
    weight_unit_mappings = {
        'g': 'gram',
        'gm': 'gram',
        'gram': 'gram',
        'kg': 'kilogram',
        'kilogram': 'kilogram',
        'mg': 'milligram',
        'milligram': 'milligram',
        'mcg': 'microgram',
        'microgram': 'microgram',
        'oz': 'ounce',
        'ounce': 'ounce',
        'lb': 'pound',
        'lbs': 'pound',
        'LBS' : 'pound',
        'pound': 'pound',
        'ton': 'ton',
        'tons': 'ton',
        'grams': 'gram',
        'kilograms': 'kilogram',
        'milligrams': 'milligram',
        'micrograms': 'microgram',
        'pounds': 'pound',
        'ounces': 'ounce',
        "g": "gram",
        "kg": "kilogram",
        "mg": "milligram",
        "oz": "ounce",
        "oz.": "ounce",
        "lb": "pound",
        "lbs": "pound",
        "t": "ton",
        "μg": "microgram",
        "mcg": "microgram"
    } 

    # List of valid weight-related units
    valid_weight_units = list(weight_unit_mappings.keys())

    # Regular expression to match numbers followed by units (ignore extra text like 'TOP')
    match = re.search(r'(\d+\.?\d*)\s*([a-zA-Z.]+)', value)
    if match:
        weight, unit = match.groups()
        unit = unit.strip().lower().replace('.', '')  # Normalize unit string by removing periods
        
        # Check if the unit is in the valid weight-related units list
        if unit in valid_weight_units:
            full_unit = weight_unit_mappings.get(unit, "")
            if full_unit:
                # If weight is 1, make sure the unit is singular
                if float(weight) == 1:
                    full_unit = full_unit.rstrip('s')  # Remove 's' for singular
                return f"{weight} {full_unit}"
    
    # If the unit is not a valid weight-related unit, return an empty string
    return ""

# Apply the cleaning function to the 'predicted_value' column
df['predicted_value_cleaned'] = df['predicted_value'].apply(clean_weight_value)

# Save the cleaned data to a new CSV file
output_file = 'cleaned_results_maximum_weight_recommendation.csv'
df.to_csv(output_file, index=False)

print(f"Cleaned data saved to {output_file}")
