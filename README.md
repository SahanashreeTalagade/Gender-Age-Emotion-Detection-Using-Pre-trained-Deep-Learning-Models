
**Gender, Age & Emotion Detection Using Pre-trained Deep Learning Models**

This project performs face detection and uses pre-trained deep learning models to predict gender, age, and emotion from facial images. It demonstrates a simple pipeline for analyzing facial attributes.

---

**Features**

* Detects faces in images
* Predicts gender (male/female)
* Estimates age range
* Recognizes emotion (happy, sad, neutral)
* Uses pre-trained CNN models

---

**Project Structure**

models/             - Pre-trained deep learning models
src/                - Source code scripts
test_image/         - Sample images for testing
LICENSE             - Project license
README.md           - Project documentation
requirements.txt    - Project dependencies

---

**Setup & Installation**

1. Clone the repository:
   git clone [https://github.com/SahanashreeTalagade/Gender-Age-Emotion-Detection-Using-Pre-trained-Deep-Learning-Models.git](https://github.com/SahanashreeTalagade/Gender-Age-Emotion-Detection-Using-Pre-trained-Deep-Learning-Models.git)
   
   cd Gender-Age-Emotion-Detection-Using-Pre-trained-Deep-Learning-Models

---

**Dependencies**

Install required Python libraries:
pip install -r requirements.txt

Dependencies include:

* opencv-python
* numpy
* tensorflow
* keras
* pillow
* matplotlib (optional)

---

**Usage**

1. Put your test images in `test_image/`.
2. Run the main script:
   python src/main.py
3. The script will detect faces and output:

   * Gender
   * Age estimate
   * Emotion
4. Results will be displayed on images with bounding boxes or printed in the console.

---

**How It Works**

1. Face Detection: Uses Haar Cascade to locate faces.
2. Attribute Prediction: Pre-trained CNN models classify gender, predict age, and recognize emotion.
3. Output: Results are displayed on images with bounding boxes and labels.

---

**Sample Images**

The `test_image/` folder contains example images to test the pipeline.

---

**License**

This project is licensed under the GPL-3.0 License.


