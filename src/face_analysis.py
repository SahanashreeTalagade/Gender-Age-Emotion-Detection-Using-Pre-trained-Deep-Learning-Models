import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import os


# Load models


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


age_net = cv2.dnn.readNetFromCaffe(
    "models/age_deploy.prototxt",
    "models/age_net.caffemodel"
)

gender_net = cv2.dnn.readNetFromCaffe(
    "models/gender_deploy.prototxt",
    "models/gender_net.caffemodel"
)

emotion_model = load_model("models/emotion_detection_model.h5")

print("✅ All models loaded successfully")


# Labels


AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)"
]

GENDER_LIST = ["Male", "Female"]

EMOTIONS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]


# Smoothing buffers


age_history = deque(maxlen=10)
gender_history = deque(maxlen=10)
emotion_history = deque(maxlen=10)


# Emotion preprocessing (SAFE)


def preprocess_emotion(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    emotion_input = normalized.reshape(1, 48, 48, 1)
    return emotion_input


# Start webcam


cap = cv2.VideoCapture(0)
print("🎥 Camera started — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)

        # AGE & GENDER 
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        gender_history.append(gender)
        gender = max(set(gender_history), key=gender_history.count)

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_BUCKETS[age_preds[0].argmax()]
        age_history.append(age)
        age = max(set(age_history), key=age_history.count)

        # EMOTION (FIXED)
        emotion_input = preprocess_emotion(face)

        # SAFETY FIX: ensure 4D input (prevents kernel error)
        if emotion_input.ndim == 5:
            emotion_input = emotion_input.reshape(1, 48, 48, 1)

        emotion_preds = emotion_model.predict(
            emotion_input, verbose=0
        )

        emotion = EMOTIONS[np.argmax(emotion_preds)]
        emotion_history.append(emotion)
        emotion = max(set(emotion_history), key=emotion_history.count)

        #DISPLAY
        label = f"{gender}, {age}, {emotion}"
        cv2.putText(
            frame, label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2
        )

    cv2.imshow("Face | Age | Gender | Emotion", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
