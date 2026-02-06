import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------
# Load face cascade
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# -------------------------------
# Load gender model
# -------------------------------
gender_net = cv2.dnn.readNetFromCaffe(
    "models/gender_deploy.prototxt",
    "models/gender_net.caffemodel"
)
GENDER_LIST = ["Male", "Female"]

# -------------------------------
# Load age model
# -------------------------------
age_net = cv2.dnn.readNetFromCaffe(
    "models/age_deploy.prototxt",
    "models/age_net.caffemodel"
)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# -------------------------------
# Load emotion model
# -------------------------------
emotion_model = load_model("models/emotion_detection_model.h5")
EMOTION_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -------------------------------
# Read input image
# -------------------------------
img = cv2.imread("test_image/baby.webp")

if img is None:
    print("❌ Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# -------------------------------
# Process each detected face
# -------------------------------
for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]

    # --- Gender preprocessing ---
    blob_gender = cv2.dnn.blobFromImage(
        face, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )
    gender_net.setInput(blob_gender)
    preds_gender = gender_net.forward()
    male_conf = preds_gender[0][0]
    female_conf = preds_gender[0][1]
    gender = "Male" if male_conf > female_conf else "Female"
    gender_conf = max(male_conf, female_conf)

    # --- Age preprocessing ---
    blob_age = cv2.dnn.blobFromImage(
        face, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )
    age_net.setInput(blob_age)
    preds_age = age_net.forward()
    age_index = preds_age[0].argmax()
    age = AGE_LIST[age_index]
    age_conf = preds_age[0][age_index]

    # --- Emotion preprocessing ---
    face_gray = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
    face_gray = face_gray.astype("float") / 255.0
    face_gray = np.expand_dims(face_gray, axis=0)
    face_gray = np.expand_dims(face_gray, axis=-1)  # add channel
    preds_emotion = emotion_model.predict(face_gray, verbose=0)
    emotion_index = preds_emotion[0].argmax()
    emotion = EMOTION_LIST[emotion_index]
    emotion_conf = preds_emotion[0][emotion_index]
   # --- Emotion post-processing to avoid false "Angry" ---
threshold = 0.75  # stricter threshold
top_indices = preds_emotion[0].argsort()[-2:][::-1]  # get top 2 predictions
top_conf = preds_emotion[0][top_indices[0]]
second_conf = preds_emotion[0][top_indices[1]]

# If top prediction is Angry but second is close OR confidence is low, set Neutral
if EMOTION_LIST[top_indices[0]] == "Angry" and (top_conf - second_conf < 0.2 or top_conf < threshold):
    emotion = "Neutral/Uncertain"
else:
    emotion = EMOTION_LIST[top_indices[0]]
emotion_conf = preds_emotion[0][top_indices[0]]  # keep confidence for label display


    # --- Draw rectangle and labels ---
label = f"{gender} ({gender_conf*100:.1f}%), Age: {age} ({age_conf*100:.1f}%), Emotion: {emotion} ({emotion_conf*100:.1f}%)"
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.putText(img, label, (x, y-10),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    

# -------------------------------
# Display final image
# -------------------------------
cv2.imshow("Gender + Age + Emotion", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
