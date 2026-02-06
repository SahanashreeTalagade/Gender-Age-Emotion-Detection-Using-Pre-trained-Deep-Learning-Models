import cv2
import os

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Create folder to save detected faces
os.makedirs("detected_faces", exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

count = 0  # Counter for saved face images

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop and save the detected face
        face_img = frame[y:y+h, x:x+w]
        count += 1
        cv2.imwrite(f"detected_faces/face_{count}.jpg", face_img)

    # Display the webcam feed
    cv2.imshow("Face Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} face images in 'detected_faces' folder.")
