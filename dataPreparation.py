import os
import cv2
import numpy as np
from sklearn.utils import shuffle

# Parameters
imgSize = 150
sequence_length = 5
folderPath = "Data"
batch_size = 1000

# Dynamically detect categories
categories = sorted(os.listdir(folderPath))
print(f"Categories detected: {categories}")

# Prepare sequences
data = []
labels = []
batch_counter = 0  # Batch counter

# Collect all sequences and labels first
all_data = []
all_labels = []

for category in categories:
    path = os.path.join(folderPath, category)
    class_num = categories.index(category)  # Numeric label for the category
    images = sorted(os.listdir(path))  # Ensure order consistency

    if len(images) < sequence_length:
        print(f"Skipping category {category}: insufficient images.")
        continue

    print(f"Processing category {category} with {len(images)} images.")

    # Create sequences
    sequences = [images[i:i + sequence_length] for i in range(0, len(images) - sequence_length + 1)]
    for seq in sequences:
        frames = []
        for img_name in seq:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            img = cv2.resize(img, (imgSize, imgSize))
            frames.append(img)
        if len(frames) == sequence_length:
            all_data.append(frames)
            all_labels.append(class_num)

# Shuffle the entire dataset
all_data, all_labels = shuffle(all_data, all_labels, random_state=42)

# Now save the data in batches, ensuring class diversity
total_sequences = len(all_data)
print(f"Total sequences: {total_sequences}")

for i in range(0, total_sequences, batch_size):
    batch_data = all_data[i:i + batch_size]
    batch_labels = all_labels[i:i + batch_size]
    batch_counter += 1
    batch_data_array = np.array(batch_data)
    batch_labels_array = np.array(batch_labels)
    print(f"Saving batch {batch_counter}... Batch data shape: {batch_data_array.shape}")
    print(f"Batch labels (unique): {np.unique(batch_labels_array)}")
    np.save(f"data_batch_{batch_counter}.npy", batch_data_array)
    np.save(f"labels_batch_{batch_counter}.npy", batch_labels_array)

print("Data preprocessing complete.")
