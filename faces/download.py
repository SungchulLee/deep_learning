import os
import urllib.request
import zipfile

from global_name_space import *

def download():
    if not os.path.exists(DOWNLOAD_DIR):
        # URL of the dataset
        url = "https://download.pytorch.org/tutorial/faces.zip"

        # Download the dataset
        zip_filename = os.path.join(DOWNLOAD_DIR, "faces.zip")
        urllib.request.urlretrieve(url, zip_filename)

        # Extract the contents of the ZIP file
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)

        # Remove the ZIP file after extraction
        os.remove(zip_filename)

    print(f"Dataset downloaded and extracted to: {DOWNLOAD_DIR}")