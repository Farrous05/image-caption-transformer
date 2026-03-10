"""
Script to download the Flickr8k dataset.
Downloads images and captions to data/flickr8k/.
"""

import os
import zipfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flickr8k")


def download_file(url, dest_path):
    """Download a file from a URL with progress."""
    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return

    print(f"  Downloading: {os.path.basename(dest_path)}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"  Downloaded: {dest_path}")


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"  Extracting: {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"  Extracted to: {extract_to}")


def download_flickr8k():
    """
    Download Flickr8k dataset.

    Note: The Flickr8k dataset is commonly hosted on Kaggle.
    You may need to download it manually from:
      https://www.kaggle.com/datasets/adityajn105/flickr8k

    After downloading, place the files in data/flickr8k/:
      - data/flickr8k/Images/        (folder with all .jpg files)
      - data/flickr8k/captions.txt   (caption annotations)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "Images"), exist_ok=True)

    images_dir = os.path.join(DATA_DIR, "Images")
    captions_file = os.path.join(DATA_DIR, "captions.txt")

    # Check if data already exists
    if os.path.exists(captions_file) and len(os.listdir(images_dir)) > 100:
        print("Flickr8k dataset already downloaded.")
        print(f"  Images: {len(os.listdir(images_dir))} files in {images_dir}")
        print(f"  Captions: {captions_file}")
        return

    print("=" * 60)
    print("  FLICKR8K DATASET DOWNLOAD")
    print("=" * 60)
    print()
    print("The Flickr8k dataset needs to be downloaded from Kaggle.")
    print()
    print("Option 1: Using Kaggle CLI (if you have it installed):")
    print("  kaggle datasets download -d adityajn105/flickr8k -p data/")
    print("  unzip data/flickr8k.zip -d data/flickr8k/")
    print()
    print("Option 2: Manual download:")
    print("  1. Go to: https://www.kaggle.com/datasets/adityajn105/flickr8k")
    print("  2. Download the dataset")
    print("  3. Extract to data/flickr8k/")
    print()
    print("Expected structure after download:")
    print("  data/flickr8k/")
    print("  ├── Images/")
    print("  │   ├── 1000268201_693b08cb0e.jpg")
    print("  │   ├── 1001773457_577c3a7d70.jpg")
    print("  │   └── ... (8091 images)")
    print("  └── captions.txt")
    print()
    print("=" * 60)


if __name__ == "__main__":
    download_flickr8k()
