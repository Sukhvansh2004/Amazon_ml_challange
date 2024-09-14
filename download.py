import os
import csv
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Create directory to save the images
output_dir = "testImages"
os.makedirs(output_dir, exist_ok=True)

# CSV file paths and the column name that contains the image links
csv_file_path = r"dataset\test.csv"
output_csv_file_path = r"dataset\testImagePath.csv"
image_column_name = "image_link"

# List to store rows for the new CSV
updated_rows = []

# Function to download an image from a URL and return the file path
def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check if the request was successful

        # Parse the URL to extract the image file name
        image_name = os.path.basename(urlparse(url).path)

        # Save the image
        image_path = os.path.join(output_dir, image_name)
        with open(image_path, 'wb') as file:
            file.write(response.content)

        print(f"Downloaded: {image_name}")
        return image_path  # Return the local image path
    except Exception as e:
        print(f"Failed to download {url}. Error: {e}")
        return None

# Function to read URLs from CSV and replace the image link with the local path
def read_and_update_csv():
    global updated_rows
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        for row in reader:
            image_url = row[image_column_name]
            image_path = download_image(image_url)  # Download and get the local path
            if image_path:
                row[image_column_name] = image_path  # Replace the URL with the local path
            updated_rows.append(row)  # Save the updated row

        return fieldnames  # Return the fieldnames to use for writing the new CSV

# Worker function for multiprocessing
def process_urls(urls):
    # Use ThreadPoolExecutor for multi-threading within each process
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_image, urls)

# Main multiprocessing executor
if __name__ == "__main__":
    # Read and download images while updating the CSV data
    fieldnames = read_and_update_csv()

    # Split URLs into chunks for multiprocessing
    image_urls = [row[image_column_name] for row in updated_rows]
    chunk_size = len(image_urls) // os.cpu_count() or 1
    url_chunks = [image_urls[i:i + chunk_size] for i in range(0, len(image_urls), chunk_size)]

    # Use ProcessPoolExecutor to parallelize across CPU cores
    with ProcessPoolExecutor() as executor:
        executor.map(process_urls, url_chunks)

    # Write the updated CSV
    with open(output_csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"All images downloaded and new CSV saved as {output_csv_file_path}.")
