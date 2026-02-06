import cv2
import numpy as np
import os

# -------------------- MODEL PATHS --------------------
AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

# -------------------- LOAD MODELS --------------------
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

# -------------------- AGE & GENDER LISTS --------------------
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# -------------------- FACE DETECTOR --------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------- WEBCAM --------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

# -------------------- SMOOTHING HISTORY --------------------
AGE_HISTORY = []
GENDER_HISTORY = []
SMOOTH_FRAMES = 20  # Number of frames to average

# Folder to save snapshots
snapshot_folder = "snapshots"
os.makedirs(snapshot_folder, exist_ok=True)
snapshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Preprocess
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # -------------------- GENDER PREDICTION --------------------
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()[0]  # shape=(2,)
        GENDER_HISTORY.append(gender_preds)
        if len(GENDER_HISTORY) > SMOOTH_FRAMES:
            GENDER_HISTORY.pop(0)
        avg_gender = np.mean(GENDER_HISTORY, axis=0)
        smoothed_gender = GENDER_LIST[np.argmax(avg_gender)]

        # -------------------- AGE PREDICTION --------------------
        age_net.setInput(blob)
        age_preds = age_net.forward()[0]  # shape=(8,)
        AGE_HISTORY.append(age_preds)
        if len(AGE_HISTORY) > SMOOTH_FRAMES:
            AGE_HISTORY.pop(0)
    # Using probabilities instead of labels allows smoothing based on model confidence, giving more stable predictions.

        avg_age = np.mean(AGE_HISTORY, axis=0)
        smoothed_age = AGE_LIST[np.argmax(avg_age)]

        # Draw rectangle and label
        label = f"{smoothed_gender}, {smoothed_age}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow("Age & Gender Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('s'):  # Save snapshot
        snapshot_count += 1
        snapshot_path = os.path.join(snapshot_folder, f"snapshot_{snapshot_count}.jpg")
        cv2.imwrite(snapshot_path, frame)
        print(f"Snapshot saved: {snapshot_path}")

cap.release()
cv2.destroyAllWindows()
