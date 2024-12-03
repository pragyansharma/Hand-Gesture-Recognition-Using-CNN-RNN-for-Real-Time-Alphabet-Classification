import numpy as np

data_batch = np.load("data_batch_final.npy")
labels_batch = np.load("labels_batch_final.npy")

print(f"Loaded data shape: {data_batch.shape}")
print(f"Loaded labels shape: {labels_batch.shape}")
print(f"Unique labels in saved batch: {np.unique(labels_batch)}")

