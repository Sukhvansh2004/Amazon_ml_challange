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
        'ml': 'millilitre', 'millilitre': 'millilitre', 'millilitres': 'millilitre',
        'cl': 'centilitre', 'centilitre': 'centilitre', 'centilitres': 'centilitre',
        'dl': 'decilitre', 'decilitre': 'decilitre', 'decilitres': 'decilitre',
        'l': 'litre', 'litre': 'liter', 'litre': 'litre', 'liter': 'litre', 'litre' : 'litre',
        'fl oz': 'fluid ounce', 'fl. oz.': 'fluid ounce', 'fluid ounce': 'fluid ounce', 'fluid ounces': 'fluid ounce', 'FL OZ': 'fluid ounce', 'FL. OZ.': 'fluid ounce', 'FL.OZ.': 'fluid ounce', 'fluid oz': 'fluid ounce',
        'cup': 'cup', 'cups': 'cup',
        'pint': 'pint', 'pints': 'pint',
        'quart': 'quart', 'quarts': 'quart',
        'gallon': 'gallon', 'gallons': 'gallon',
        'imperial gallon': 'imperial gallon', 'imperial gallons': 'imperial gallon',
        'oz': 'ounce', 'ounce': 'ounce', 'ounces': 'ounce',
        'cubic foot': 'cubic foot', 'cubic feet': 'cubic foot',
        'cubic inch': 'cubic inch', 'cubic inches': 'cubic inch',
        "fl": "fluid ounce",
        "l": "liter",
        "ml": "milliliter",
        "cl": "centiliter",
        "fl oz": "fluid ounce",
        "fl. oz.": "fluid ounce",
        "fl.": "fluid ounce",
        "fluid": "fluid ounce",
        "gal": "gallon",
        "pt": "pint",
        "qt": "quart",
        "imp gal": "imperial gallon",
        "c": "cup",
        "dl": "deciliter"
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
