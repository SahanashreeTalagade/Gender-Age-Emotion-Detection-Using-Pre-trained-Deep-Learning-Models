import os
print("Running from:", os.getcwd())

from tensorflow.keras.models import load_model
import numpy as np

try:
    # Load the emotion model
    model = load_model("models/emotion_detection_model.h5")
    print("✅ Emotion model loaded successfully!")

    # Print model summary
    model.summary()

    # Create a dummy input (FER2013 format: 48x48 grayscale)
    dummy_input = np.random.rand(1, 48, 48, 1)

    # Run a test prediction
    prediction = model.predict(dummy_input, verbose=0)

    print("✅ Prediction successful!")
    print("Prediction shape:", prediction.shape)

except Exception as e:
    print("❌ Error loading emotion model:")
    print(e)
