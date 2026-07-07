# Deep Learning based Face Recognition System on Raspberry Pi 5

An edge AI-based face recognition system designed for **real-time identity recognition on Raspberry Pi 5**. The system combines **MTCNN** for face detection with a **fine-tuned ResNet-18** model for face recognition, enabling efficient on-device inference without relying on cloud services.

## Overview

This project implements a highly accurate face recognition pipeline optimized for edge deployment. The system captures images using a camera connected to the Raspberry Pi, detects faces using **MTCNN**, generates facial embeddings through a fine-tuned **ResNet-18**, and performs identity recognition by comparing embeddings with a precomputed face database.

The model was trained on a custom facial image dataset and achieved a **98.04% test accuracy**, making it suitable for real-time embedded AI applications such as access control, attendance monitoring and smart surveillance.

---

## Features

* Real-time face detection using **MTCNN**
* Face recognition using a fine-tuned **ResNet-18**
* On-device inference on Raspberry Pi 5
* Facial embedding generation and similarity matching
* Bounding box visualization with predicted identity

---

## System Architecture

```text
Camera
   │
   ▼
Image Capture
   │
   ▼
Face Detection (MTCNN)
   │
   ▼
Face Alignment & Preprocessing
   │
   ▼
Fine-Tuned ResNet-18
   │
   ▼
Facial Embedding Generation
   │
   ▼
Embedding Similarity Matching
   │
   ▼
Predicted Output
```

---

## Tech Stack

| Category                | Technologies         |
| ----------------------- | -------------------- |
| Programming Language    | Python               |
| Deep Learning Framework | PyTorch              |
| Face Detection          | MTCNN                |
| Deep Learning Model     | Fine-Tuned ResNet-18 |
| Computer Vision         | OpenCV               |
| Hardware                | Raspberry Pi 5       |
| Operating System        | Raspberry Pi OS      |

---

## Project Structure

```text
face-recognition-webapp/
│
├── model/
│   └── face_recognition_model.pth
├── static/
│   └── script.js
|   └── style.css
├── templates/
│   └── index.html
├── README.md
├── app.py
├── embeddings.pt
└── requirements.txt
```

---

## Dataset

Our dataset consists of approximately 5,280 face images without augmentation, sourced from 50 facial video recordings of our friends. To process this data, we first utilized the OpenCV library to extract raw snapshots from the video footage. Subsequently, the Multi-task Cascaded Convolutional Networks (MTCNN) was used to detect faces and crop the images to isolate the region of interest from the extracted frames. The final processed dataset was partitioned in the ratio 80:20, 80% images allocated for training and 20% for testing.
The dataset used for training and evaluation is **not included** in this repository due to **Size and Privacy consideration**.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/ArindamECB22/face-recognition-webapp.git
cd face-recognition-webapp
```

Create a Python Virtual Environment

Open the Raspberry Pi terminal and navigate to the project directory.

Create a virtual environment named faceenv:
```bash
python3 -m venv faceenv
```
Activate the virtual environment:
```bash
source faceenv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

To run the application, run this command in the Raspberry Pi terminal:
```bash
python3 app.py
```
Make sure you are in the Project directory & the virtual environment is active.

It will show a link to the web app. For e.g. * Running on http://0.0.0.0:5000

Copy & Paste the link in a browser (Chrome recommended). 
This will open the web app.

The Raspberry Pi is now ready to:

1. Capture frames from the connected camera.
2. Detect faces using **MTCNN**.
3. Generate facial embeddings using the fine-tuned **ResNet-18**.
4. Compare embeddings with the stored face database.
5. Display the recognized identity with a bounding box.

---

## Model Performance

| Metric              |          Value |
| ------------------- | -------------: |
| Test Accuracy       |     **98.04%** |
| Backbone Network    |      ResNet-18 |
| Face Detection      |          MTCNN |
| Deployment Platform | Raspberry Pi 5 |

---

## Applications

* Smart attendance systems
* Intelligent access control
* Smart home security
* Visitor identification
* Edge AI surveillance

---

## Author

**Arindam**

GitHub: https://github.com/ArindamECB22

LinkedIn: https://www.linkedin.com/in/arindam-s-613180273

---

## License

This project is intended for educational and research purposes.
