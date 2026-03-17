import os
import shutil
import random

# Source dataset folder
source_dir = "images"

# Output dataset folder
output_dir = "bird-dataset"

categories = {
    "fly": ["Albatross", "Crow", "Sparrow", "Eagle", "Hawk"],
    "no_fly": ["Ostrich", "Penguin"],
    "fly_swim": ["Duck", "Swan", "Pelican", "Seagull"]
}

# Create dataset folders
for split in ["train", "val"]:
    for category in categories.keys():
        os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

# Convert dataset
for folder in os.listdir(source_dir):

    for category, birds in categories.items():

        if any(bird.lower() in folder.lower() for bird in birds):

            folder_path = os.path.join(source_dir, folder)

            images = os.listdir(folder_path)
            random.shuffle(images)

            split = int(0.8 * len(images))

            train_imgs = images[:split]
            val_imgs = images[split:]

            for img in train_imgs:
                shutil.copy(
                    os.path.join(folder_path, img),
                    os.path.join(output_dir, "train", category, img)
                )

            for img in val_imgs:
                shutil.copy(
                    os.path.join(folder_path, img),
                    os.path.join(output_dir, "val", category, img)
                )

print("Dataset conversion completed!")