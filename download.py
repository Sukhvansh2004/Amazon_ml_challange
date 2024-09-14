import os
import csv
import requests
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor

# Create directories to save the images and failed download log
output_dir = "testImages"
os.makedirs(output_dir, exist_ok=True)

# CSV file paths and the column name that contains the image links
csv_file_path = r"dataset\test.csv"
output_csv_file_path = r"dataset\testImagePath.csv"
failed_csv_file_path = r"dataset\failed_downloads.csv"
image_column_name = "image_link"

# List to store rows for the new CSV and failed downloads
updated_rows = []
failed_rows = []

# Function to download an image from a URL and return the file path
def download_image(url):
    try:
        # Parse the URL to extract the image file name
        image_name = os.path.basename(urlparse(url).path)
        image_path = os.path.join(output_dir, image_name)

        # Check if the image already exists
        if os.path.exists(image_path):
            return image_path  # Skip downloading and return the existing path

        # Download the image if not already downloaded
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check if the request was successful

        # Save the image
        with open(image_path, 'wb') as file:
            file.write(response.content)

        return image_path  # Return the local image path
    except Exception as e:
        # print(f"Failed to download {url}. Error: {e}")
        return None

# Function to process a batch of URLs and update the corresponding rows
def process_url_chunk(url_chunk):
    processed_rows = []
    failed_rows_chunk = []
    for row in url_chunk:
        image_url = row[image_column_name]
        image_path = download_image(image_url)  # Download and get the local path or check if it exists
        if image_path:
            row[image_column_name] = image_path  # Replace the URL with the local path
        else:
            failed_rows_chunk.append(row)  # Append the failed download row
        processed_rows.append(row)  # Save the updated row
    return processed_rows, failed_rows_chunk

# Function to read URLs from CSV and split data into chunks for multiprocessing
def read_csv_in_chunks():
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        # Read all rows and split them into chunks for parallel processing
        rows = list(reader)
        chunk_size = len(rows) // os.cpu_count() or 1
        url_chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
        
        return fieldnames, url_chunks

# Main multiprocessing executor
if __name__ == "__main__":
    # Read and split data into chunks
    fieldnames, url_chunks = read_csv_in_chunks()

    # Use ProcessPoolExecutor to parallelize the work across CPU cores
    with ProcessPoolExecutor() as process_executor:
        # Process the chunks in parallel
        results = process_executor.map(process_url_chunk, url_chunks)

    # Collect all updated rows and failed downloads from the processed chunks
    for result_chunk, failed_chunk in results:
        updated_rows.extend(result_chunk)
        failed_rows.extend(failed_chunk)

    # Write the updated CSV with successfully downloaded image paths
    with open(output_csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    # Write the failed downloads CSV
    if failed_rows:
        with open(failed_csv_file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(failed_rows)
        print(f"Failed downloads saved as {failed_csv_file_path}.")
    else:
        print("No failed downloads.")

    print(f"All images downloaded and new CSV saved as {output_csv_file_path}.")
