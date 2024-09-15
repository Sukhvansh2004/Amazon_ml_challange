import pandas as pd
import os

#Load the CSV file
df = pd.read_csv('test.csv')

#Extract unique entity names from the CSV
unique_entities = df['entity_name'].unique()

#Create a folder to store entity-specific CSV files if it doesn't exist
if not os.path.exists('entity_csvs'):
    os.makedirs('entity_csvs')

#Create individual CSV files for each entity_name
for entity_name in unique_entities:
    # Filter the dataframe for the current entity_name
    entity_df = df[df['entity_name'] == entity_name]

#Save to a new CSV file
    output_filename = os.path.join('entity_csvs', f'{entity_name}.csv')
    entity_df.to_csv(output_filename, index=True)
    print(f"Created CSV for entity '{entity_name}' at {output_filename}")
