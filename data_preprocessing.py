# src/data_preprocessing.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define dataset directories
with_mask_dir = '../dataset/with_mask'
without_mask_dir = '../dataset/without_mask'

# Initialize lists to hold image data and labels
data = []
labels = []

# Load images from 'with_mask'
for img_name in os.listdir(with_mask_dir):
    img_path = os.path.join(with_mask_dir, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # normalize
        data.append(img)
        labels.append(1)  # 1 = with_mask

# Load images from 'without_mask'
for img_name in os.listdir(without_mask_dir):
    img_path = os.path.join(without_mask_dir, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # normalize
        data.append(img)
        labels.append(0)  # 0 = without_mask

# Convert lists to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, shuffle=True
)

# Print shapes to verify
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Optional: Save to file for reuse
# np.save("../models/X_train.npy", X_train)
# np.save("../models/X_test.npy", X_test)
# np.save("../models/y_train.npy", y_train)
# np.save("../models/y_test.npy", y_test)
