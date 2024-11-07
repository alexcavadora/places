import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class BeachMountainDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        self.images = []
        self.labels = []

        beach_dir = os.path.join(img_dir, 'beach')
        for img_name in os.listdir(beach_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(beach_dir, img_name))
                self.labels.append(0)  # 0 for beach

        # Load mountain images
        mountain_dir = os.path.join(img_dir, 'mountain')
        for img_name in os.listdir(mountain_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(mountain_dir, img_name))
                self.labels.append(1)  # 1 for mountain

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
