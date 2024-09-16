import pandas as pd

# Load the CSV file
file_path = 'cleaned_results_item_volume.csv'  # Adjust the path as necessary
df = pd.read_csv(file_path)

# Display the first few rows to inspect the data
print(df.head())
print(df.columns)
