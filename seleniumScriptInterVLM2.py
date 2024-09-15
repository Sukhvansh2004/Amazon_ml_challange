import os
import time
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from multiprocessing import Pool
from threading import Thread

# Function to run inference on a single image and text prompt
def run_inference_on_image_and_text(image_path, text_prompt, driver):
    try:
        # Navigate to the website
        driver.get("https://internvl.opengvlab.com/")

        # Wait for page to load (you can add more explicit waits if necessary)
        time.sleep(2)

        # Find the upload button and upload the image
        upload_button = driver.find_element(By.XPATH, "//input[@type='file']")
        upload_button.send_keys(image_path)

        # Enter the text prompt in the input field (modify based on actual UI)
        text_input = driver.find_element(By.XPATH, "//textarea[@id='prompt']")  # Modify the XPath as needed
        text_input.send_keys(text_prompt)

        # Trigger the inference (assuming there is a button to start inference)
        start_inference_button = driver.find_element(By.XPATH, "//button[text()='Run Inference']")
        start_inference_button.click()

        # Wait for inference to complete and results to appear
        time.sleep(5)  # Adjust according to the website's response time

        # Assuming there's some result element to retrieve (modify based on actual UI)
        result = driver.find_element(By.XPATH, "//div[@id='result']")  # Placeholder, modify as needed
        return result.text

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Thread function to handle multiple images and text prompts
def threaded_inference(image_paths, text_prompts, output_file):
    # Setup the WebDriver (this should be within the thread)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    results = []
    for image_path, text_prompt in zip(image_paths, text_prompts):
        result = run_inference_on_image_and_text(image_path, text_prompt, driver)
        if result:
            results.append([image_path, text_prompt, result])

    # Save results to the CSV file
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)

    # Close the driver after all images are processed
    driver.quit()

# Function for multiprocessing
def process_batch(image_batch, text_batch, output_file):
    # Use threading within each process
    num_threads = 1 #min(5, len(image_batch))  # Adjust number of threads based on system resources
    chunk_size = len(image_batch) // num_threads
    threads = []

    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_threads - 1 else len(image_batch)
        thread = Thread(target=threaded_inference, args=(image_batch[start_idx:end_idx], text_batch[start_idx:end_idx], output_file))
        thread.start()
        threads.append(thread)

    # Join all threads
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    # Get all image paths from the test folder
    image_folder = r"C:\Users\Sukhvansh Jain\Documents\Projects and Competitions\Amazon ML Challenge\PS\Amazon_ml_challange\images"
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Define corresponding text prompts for each image
    # Example prompts; replace with actual text inputs
    text_prompts = ["Describe the image", "What is happening here?", "What objects are in the image?"] * (len(images) // 3 + 1)
    text_prompts = text_prompts[:len(images)]  # Ensure the number of prompts matches the number of images

    # Define the output CSV file
    output_file = "inference_results.csv"

    # Create the CSV file and write the header
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Text Prompt", "Inference Result"])

    # Define the number of processes (adjust based on your system's CPU)
    num_processes = 1 #os.cpu_count()
    batch_size = len(images) // num_processes

    # Create batches of images and corresponding text prompts for each process
    image_batches = [images[i * batch_size:(i + 1) * batch_size] for i in range(num_processes)]
    text_batches = [text_prompts[i * batch_size:(i + 1) * batch_size] for i in range(num_processes)]

    # Use multiprocessing to handle image batches and text prompts
    with Pool(num_processes) as pool:
        pool.starmap(process_batch, zip(image_batches, text_batches, [output_file] * num_processes))
