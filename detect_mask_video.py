import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('../models/mask_detector_model.h5')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)
        prediction = model.predict(face_input)[0][0]
        label = "Mask" if prediction >= 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow('Mask Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
