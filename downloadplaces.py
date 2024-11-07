import os
import torch
from torch.utils.data import Dataset
from torchvision import models
import requests

def download_places365_files():
    # Download model weights if not present
    weights_url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
    weights_path = 'resnet50_places365.pth.tar'

    if not os.path.exists(weights_path):
        print(f"Downloading Places365 weights...")
        response = requests.get(weights_url)
        with open(weights_path, 'wb') as f:
            f.write(response.content)
        print("Download complete!")

    # Download categories file if not present
    categories_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    categories_path = 'categories_places365.txt'

    if not os.path.exists(categories_path):
        print(f"Downloading Categories file...")
        response = requests.get(categories_url)
        with open(categories_path, 'wb') as f:
            f.write(response.content)
        print("Download complete!")

if __name__ == "__main__":
    download_places365_files()
