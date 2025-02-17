import albumentations as A
import cv2
import os
import numpy as np
from tqdm import tqdm

# Augmentation Pipeline
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.CLAHE(p=0.3),
])

# Directories
DATA_DIR = "C:/Users/HP/OneDrive/Desktop/DermDetect/data"
IMG_DIR_1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")
OUTPUT_DIR = os.path.join(DATA_DIR, "HAM10000_augmented")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process images
for folder in [IMG_DIR_1, IMG_DIR_2]:
    for img_name in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = augmentation(image=img)["image"]
        
        # Save augmented image
        aug_path = os.path.join(OUTPUT_DIR, f"aug_{img_name}")
        cv2.imwrite(aug_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

print("✅ Data Augmentation Complete!")
