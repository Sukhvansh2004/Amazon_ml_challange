import pandas as pd
import re

# Load the CSV file
file_path = 'results_wattage.csv'
df = pd.read_csv(file_path)

# Function to expand wattage units and handle singular/plural forms
def expand_wattage_units(predicted_value):
    # Check if the value is a string and not empty
    if isinstance(predicted_value, str):
        # Regular expression to identify wattage values with singular/plural units
        match = re.match(r'(\d+(\.\d+)?)(\s*)(watts?|kilowatts?|W|KW)', predicted_value, re.IGNORECASE)
        if match:
            value = match.group(1)  # the numeric part
            unit = match.group(4).lower()  # the unit part in lowercase
            
            # Map abbreviations and plurals to singular full forms for wattage
            unit_map = {
                'w': 'watt',
                'kw': 'kilowatt',
                'watt': 'watt',
                'watts': 'watt',
                'kilowatt': 'kilowatt',
                'kilowatts': 'kilowatt'
            }
            
            # Ensure the unit is converted to its singular full form
            full_form_unit = unit_map.get(unit, unit)
            
            # Return the value with the singular full-form unit
            return f"{value} {full_form_unit}"
    
    # Return empty string if no match or incorrect format
    return ''

# Apply the function to the 'predicted_value' column
df['expanded_value'] = df['predicted_value'].apply(expand_wattage_units)

# Save the updated DataFrame to a new CSV file
df.to_csv('cleaned_results_wattage_values.csv', index=False)
print("Saved cleaned file")