import pandas as pd

# Specify the directory containing the CSV files and the file pattern
csv_files = ['cleaned_results_item_volume.csv', 'cleaned_results_item_weight.csv', 'cleaned_results_maximum_weight_recommendation.csv', 'cleaned_results_voltage_values.csv', 'cleaned_results_wattage_values.csv']

# Initialize an empty list to store individual DataFrames
df_list = []

# Read and append each CSV file into the list
for file in csv_files:
    df = pd.read_csv(file)
    
    # Convert the 'index' column to numeric, invalid parsing will be set as NaN
    df['index'] = pd.to_numeric(df['index'], errors='coerce')
    
    # Drop rows where 'index' is NaN (garbage or non-integer values)
    df = df.dropna(subset=['index'])
    
    # Convert the 'index' column to integer after dropping invalid values
    df['index'] = df['index'].round().astype(int)
    
    df_list.append(df)

# Concatenate all DataFrames in the list
combined_df = pd.concat(df_list)

# Sort the combined DataFrame by the 'index' column
combined_df = combined_df.sort_values(by='index').reset_index(drop=True)

# Save the combined DataFrame to a new CSV file
combined_df[["index", "prediction"]].to_csv('combined_output.csv', index=False)

print("CSV files combined and saved to 'combined_output.csv'.")
