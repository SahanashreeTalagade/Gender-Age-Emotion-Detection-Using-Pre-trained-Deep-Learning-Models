# Gender-Age-Emotion-Detection-Using-Pre-trained-Deep-Learning-Models
This project performs **face detection** and uses **pre-trained deep learning models** to predict **gender**, **age**, and **emotion** from facial images. It provides a simple pipeline that demonstrates how deep learning and computer vision can analyze facial attributes in static images.

---

##  Features

- Detects faces in images
- Predicts **gender** (e.g., male/female)
- Estimates **age range**
- Recognizes **emotion** (e.g., happy, sad, neutral)
- Uses **pre-trained CNN models** for inference

---

##  Project Structure
- models/ # Pre-trained deep learning models
- src/ # Source code scripts
- test_image/ # Sample images for testing
- LICENSE # Project license
-  README.md # Project documentation

------
## 🚀 Setup & Installation

1. **Clone the repository**

```bash
git clone https://github.com/SahanashreeTalagade/Gender-Age-Emotion-Detection-Using-Pre-trained-Deep-Learning-Models.git
cd Gender-Age-Emotion-Detection-Using-Pre-trained-Deep-Learning-Models

-----
Dependencies

Add a section explaining the Python libraries your project uses, and point to requirements.txt. For example:

##  Dependencies

Install required Python libraries using:

```bash
pip install -r requirements.txt


Dependencies include:

opencv-python

numpy

tensorflow

keras

pillow

matplotlib (optional)


---

### 2️ Usage Instructions

Explain **how to actually run your code**. For example:

```markdown
##  Usage

1. Put your test images in `test_image/` folder.
2. Run the main script:

```bash
python src/main.py


The script will detect faces and output:

Gender

Age estimate

Emotion

Results may be displayed on images or printed in the console.


---

### 3️ How It Works (Optional but nice)

Give a short explanation of the pipeline:

```markdown
##  How It Works

1. **Face Detection:** Uses Haar Cascade to locate faces.
2. **Attribute Prediction:** Pre-trained CNN models classify gender, predict age, and recognize emotion.
3. **Output:** Shows results on images with bounding boxes and labels.




