import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_detection_model.h5")

print("Model path:", MODEL_PATH)
print("Exists:", os.path.exists(MODEL_PATH))

emotion_model = load_model(MODEL_PATH, compile=False)
print("Emotion model loaded successfully ✅")
