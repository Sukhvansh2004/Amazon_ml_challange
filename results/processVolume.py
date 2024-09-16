import pandas as pd
import re

# Load the CSV file
file_path = 'results_item_volume.csv'  # Replace with the correct path to your file
df = pd.read_csv(file_path)

# Function to extract and clean the first valid value with full unit expansion
def clean_volume_value(value):
    if pd.isna(value):
        return ""  # Return empty string for NaN values
    
    # Regular expressions for volume units and their full forms (case-insensitive)
    unit_mappings = {
        'ml': 'milliliters', 'millilitre': 'milliliters', 'millilitres': 'milliliters',
        'cl': 'centiliters', 'centilitre': 'centiliters', 'centilitres': 'centiliters',
        'dl': 'deciliters', 'decilitre': 'deciliters', 'decilitres': 'deciliters',
        'l': 'liters', 'litre': 'liters', 'liters': 'liters', 'liter': 'liters',
        'fl oz': 'fluid ounces', 'fl. oz.': 'fluid ounces', 'fluid ounce': 'fluid ounces', 'fluid ounces': 'fluid ounces', 'FL OZ': 'fluid ounces', 'FL. OZ.': 'fluid ounces',
        'cup': 'cups', 'cups': 'cups',
        'pint': 'pints', 'pints': 'pints',
        'quart': 'quarts', 'quarts': 'quarts',
        'gallon': 'gallons', 'gallons': 'gallons',
        'imperial gallon': 'imperial gallons', 'imperial gallons': 'imperial gallons',
        'oz': 'ounces', 'ounce': 'ounces', 'ounces': 'ounces',
        'cubic foot': 'cubic feet', 'cubic feet': 'cubic feet',
        'cubic inch': 'cubic inches', 'cubic inches': 'cubic inches'
    }
    
    # Regular expression to match numbers followed by units (accounting for periods and spaces)
    match = re.search(r'(\d+\.?\d*)\s*([a-zA-Z.\s]+)', value)
    if match:
        volume, unit = match.groups()
        unit = unit.strip().lower().replace('.', '')  # Normalize unit string by removing periods
        
        # Check if the unit is singular or plural and map to its full form
        full_unit = unit_mappings.get(unit, "")
        if full_unit:
            # If volume is 1, make sure the unit is singular
            if float(volume) == 1:
                full_unit = full_unit.rstrip('s')  # Remove 's' for singular
            return f"{volume} {full_unit}"
    
    return ""  # If no valid value or unit is found

# Apply the cleaning function to the 'predicted_value' column
df['predicted_value_cleaned'] = df['predicted_value'].apply(clean_volume_value)

# Save the cleaned data to a new CSV file
output_file = 'cleaned_results_item_volume.csv'
df.to_csv(output_file, index=False)

print(f"Cleaned data saved to {output_file}")
