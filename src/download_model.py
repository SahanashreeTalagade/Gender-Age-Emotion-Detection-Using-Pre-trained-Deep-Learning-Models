from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Hammad712/Emotion_Detection",
    filename="emotion_detection_model.h5"
)

print("Downloaded to:", model_path)
