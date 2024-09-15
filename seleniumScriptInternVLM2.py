import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

# Set up paths
input_csv = r"dataset\sample_test_mini.csv"
output_csv = r"dataset\output_selenium.csv"
imageFolder = r"images"

# Read the input CSV (image URL, ID, and prompt)
df = pd.read_csv(input_csv, header=None, names=["index", "image_url", "group_id", "entity_name"])

# Function to download images locally from URLs
def download_image(image_url):
    image_name = image_url.split('/')[-1]
    image_path = os.path.join(imageFolder, image_name)
    if not os.path.exists(image_path):
        import requests
        response = requests.get(image_url)
        with open(image_path, 'wb') as f:
            f.write(response.content)
    return image_path

# Function to process each image and prompt
def process_image(image_url, attribute):
    image_name = image_url.split('/')[-1]
    image_path = download_image(image_url)  # Download image
    
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    prompt = f"What is the {attribute} of the product in the given image?"
    
    try:
        # Navigate to InternVL2 website
        driver.get('https://internvl.opengvlab.com/')
        
        # Wait until the image upload button is available
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "file-upload-input")))
        
        # Upload the image
        upload_button = driver.find_element(By.ID, "file-upload-input")
        upload_button.send_keys(image_path)

        # Input the prompt
        prompt_input = driver.find_element(By.ID, "user-question-input")  # Prompt input field ID
        prompt_input.clear()
        prompt_input.send_keys(prompt)

        # Click the submit button to run inference
        submit_button = driver.find_element(By.XPATH, "//button[text()='Run']")  # Corrected XPath for the "Run" button
        submit_button.click()

        # Wait for the output to appear
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//div[@class='stMarkdown']//p")))  
        output = driver.find_element(By.XPATH, "//div[@class='stMarkdown']//p").text  # Extracting the result

        return image_name, prompt, output
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return image_name, prompt, None
    finally:
        driver.quit()

# Function to process all images using multithreading
def process_all_images():
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_image, row['image_url'], row['entity_name']) for _, row in df.iterrows()]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
    
    # Save the results to a new CSV file
    result_df = pd.DataFrame(results, columns=["image_name", "prompt", "output"])
    result_df.to_csv(output_csv, index=False)

# Main function
if __name__ == "__main__":
    process_all_images()
