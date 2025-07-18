# train_model.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Load dataset
with_mask_dir = '../dataset/with_mask'
without_mask_dir = '../dataset/without_mask'

data = []
labels = []

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

# Load with_mask images
for img_name in os.listdir(with_mask_dir):
    if os.path.splitext(img_name)[1].lower() not in IMAGE_EXTENSIONS:
        continue
    img = cv2.imread(os.path.join(with_mask_dir, img_name))
    if img is not None:
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        data.append(img)
        labels.append(1)

# Load without_mask images
for img_name in os.listdir(without_mask_dir):
    if os.path.splitext(img_name)[1].lower() not in IMAGE_EXTENSIONS:
        continue
    img = cv2.imread(os.path.join(without_mask_dir, img_name))
    if img is not None:
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        data.append(img)
        labels.append(0)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 2. Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# 3. Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Train model
model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_test, y_test))

# 5. Save model to models folder
model.save('../models/mask_detector_model.h5')
print("Model saved to ../models/mask_detector_model.h5")
