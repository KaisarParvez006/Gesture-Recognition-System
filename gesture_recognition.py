import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained CNN model (dummy or basic model)
model = load_model('gesture_model.h5')
classes = ['Palm', 'Fist', 'Thumbs Up', 'Thumbs Down', 'Okay', 'Peace']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI)
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    # Preprocess ROI for model
    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.reshape(roi_normalized, (1, 64, 64, 3))

    # Predict gesture
    prediction = model.predict(roi_reshaped)
    class_id = np.argmax(prediction)
    gesture = classes[class_id]

    # Display result
    cv2.putText(frame, gesture, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()