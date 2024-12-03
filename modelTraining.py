import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     TimeDistributed, LSTM)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
import glob
import matplotlib.pyplot as plt
import json

# Parameters
sequence_length, img_height, img_width, channels = 5, 150, 150, 3

# Load preprocessed data in batches
data_files = sorted(glob.glob("data_batch_*.npy"))
label_files = sorted(glob.glob("labels_batch_*.npy"))

# Load and concatenate data and labels
data_list = [np.load(file) for file in data_files]
labels_list = [np.load(file) for file in label_files]

data = np.concatenate(data_list, axis=0)
labels = np.concatenate(labels_list, axis=0)

# Normalize data
data = data.astype('float32') / 255.0

# Debug prints to verify data
print(f"Loaded data shape: {data.shape}")  # Expected shape: (total_samples, sequence_length, img_height, img_width, channels)
print(f"Loaded labels shape: {labels.shape}")  # Expected shape: (total_samples,)
print(f"Loaded labels unique values: {np.unique(labels)}")

# Convert labels to integers
labels = labels.astype(int)

# Check label distribution
unique_labels, label_counts = np.unique(labels, return_counts=True)
print("Label distribution:")
for label, count in zip(unique_labels, label_counts):
    print(f"Label {label}: {count} samples")

# Dynamically determine the number of classes
num_classes = int(np.max(labels) + 1)
print("Number of classes:", num_classes)

if num_classes != 26:
    print(f"Warning: Expected 26 classes, but found {num_classes}. Check your data.")

# Convert labels to one-hot encoding
labels_categorical = to_categorical(labels, num_classes=num_classes)

# Shuffle data and labels together
data, labels_categorical = shuffle(data, labels_categorical, random_state=42)

# Use StratifiedShuffleSplit to maintain class distribution in both sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(data, np.argmax(labels_categorical, axis=1)):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels_categorical[train_index], labels_categorical[test_index]

# Validation dataset insight
val_labels = np.argmax(y_test, axis=1)
print(f"Validation labels unique values: {np.unique(val_labels)}")
print(f"Validation labels distribution: {np.bincount(val_labels)}")

# Define categories (A-Z)
categories = [chr(i) for i in range(65, 65 + num_classes)]
print("Categories:", categories)

# Build the CNN-RNN model
model = Sequential()

# CNN layers for each frame
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'),
                          input_shape=(sequence_length, img_height, img_width, channels)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))

# RNN layers
model.add(LSTM(128, activation='tanh', return_sequences=False))

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model summary to a text file
with open("model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("Model summary saved to 'model_summary.txt'")

# Implement callbacks for early stopping and model checkpointing
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_gesture_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint]
)
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# Save the final model
model.save("gesture_model.h5")
print("Model training complete.")

# Save training history
with open("training_history.json", "w") as f:
    json.dump(history.history, f)
print("Training history saved to 'training_history.json'.")

# Plot training and validation accuracy and loss
# Accuracy plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.show()

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history.get('val_loss', []), label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
plt.show()

# Testing the model on a sample from the test set
test_sequence = X_test[0:1]  # Take the first sequence from the test set
prediction = model.predict(test_sequence)
predicted_index = np.argmax(prediction)
true_index = np.argmax(y_test[0])

if predicted_index < len(categories):
    predicted_label = categories[predicted_index]
    true_label = categories[true_index]
    print(f"Test Prediction: {predicted_label}, True Label: {true_label}")
else:
    print(f"Error: Predicted index {predicted_index} is out of range for categories.")
