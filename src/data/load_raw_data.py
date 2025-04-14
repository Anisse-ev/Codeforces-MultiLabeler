import os
import gdown
import zipfile
import shutil
from config.links_and_paths import RAW_DATA_LINK, RAW_DATA_DIR

def download_and_extract_json_files(RAW_DATA_LINK, RAW_DATA_DIR):
    # Extract parent directory and target folder name
    parent_dir = os.path.dirname(RAW_DATA_DIR)

    # Get the file ID from the Google Drive link
    file_id = RAW_DATA_LINK.split('/d/')[1].split('/')[0]
    file_url = f'https://drive.google.com/uc?export=download&id={file_id}'

    # Create a temporary path to download and extract the zip
    tmp_zip_path = os.path.join(parent_dir, 'temp_data.zip')
    tmp_extract_path = os.path.join(parent_dir, 'temp_extracted')

    # Clean previous target directory if exists
    if os.path.exists(RAW_DATA_DIR):
        shutil.rmtree(RAW_DATA_DIR)
        print(f"Removed existing directory: {RAW_DATA_DIR}")

    # Clean any previous temp
    if os.path.exists(tmp_extract_path):
        shutil.rmtree(tmp_extract_path)

    os.makedirs(tmp_extract_path)

    # Download the ZIP file
    print("Downloading the zip file...")
    gdown.download(file_url, tmp_zip_path, quiet=False)

    # Extract it into the temp location
    print("Extracting...")
    with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_extract_path)

    # Remove the zip file
    os.remove(tmp_zip_path)

    # Get the single folder that was extracted
    extracted_items = os.listdir(tmp_extract_path)
    assert len(extracted_items) == 1, "Expected only one folder inside the zip."
    extracted_folder_path = os.path.join(tmp_extract_path, extracted_items[0])

    # Move and rename it to the target RAW_DATA_DIR
    shutil.move(extracted_folder_path, RAW_DATA_DIR)
    shutil.rmtree(tmp_extract_path)

    print(f"Extracted and renamed to: {RAW_DATA_DIR}")

if __name__ == "__main__":
    
    # Call the function to import data
    download_and_extract_json_files(RAW_DATA_LINK, RAW_DATA_DIR)
