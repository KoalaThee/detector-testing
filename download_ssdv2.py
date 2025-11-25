#!/usr/bin/env python3
"""
Helper script to download SSD MobileNet V2 model for OpenCV
"""

import os
import urllib.request
import tarfile

print("=== SSD MobileNet V2 Model Downloader ===\n")

# Model files
MODEL_URL = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

MODEL_TAR = "ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
MODEL_DIR = "ssd_mobilenet_v2_coco_2018_03_29"
CONFIG_FILE = "ssd_mobilenet_v2_coco.pbtxt"

# Download model
if not os.path.exists(MODEL_DIR):
    print(f"Downloading model from {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_TAR)
        print("✓ Downloaded model archive")
        
        print("Extracting model files...")
        with tarfile.open(MODEL_TAR, 'r:gz') as tar:
            tar.extractall()
        print(f"✓ Extracted to {MODEL_DIR}/")
        
        # Clean up tar file
        os.remove(MODEL_TAR)
        print("✓ Cleaned up archive file")
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        exit(1)
else:
    print(f"✓ Model directory already exists: {MODEL_DIR}/")

# Download config
if not os.path.exists(CONFIG_FILE):
    print(f"\nDownloading config from {CONFIG_URL}...")
    try:
        urllib.request.urlretrieve(CONFIG_URL, CONFIG_FILE)
        print(f"✓ Downloaded {CONFIG_FILE}")
    except Exception as e:
        print(f"✗ Error downloading config: {e}")
        exit(1)
else:
    print(f"✓ Config file already exists: {CONFIG_FILE}")

print("\n=== Setup Complete! ===")
print("\nYou can now run:")
print("  python detector-testing/test_ssd.py --video <your_video> --all-boxmot")
