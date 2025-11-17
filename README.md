🛡️ Smart Surveillance System using Deep CNN-Based Face Recognition
A Real-Time Smart Surveillance Framework using Convolutional Neural Networks (CNNs), Face Embeddings, and Deep Learning–Based Identification

This system automatically detects and recognizes individuals using FaceNet embeddings + MTCNN face detection. It compares live camera frames against a pre-enrolled database of known users. If an unknown person appears, the system captures the face and sends email alerts, triggers logging, and updates the surveillance UI.

This implementation follows the methodology described in your research paper:

✔ Dataset Support: LFW, VGGFace2, CelebA, Custom dataset
✔ Face Detection: MTCNN
✔ Face Embeddings: FaceNet (512-D feature vectors)
✔ Similarity Matching: Cosine + FAISS
✔ Real-Time Processing: OpenCV
✔ Application Layer: Python Flask (API) + Node.js (Web UI)
✔ Optional: Emotion Recognition using CNN (FER2013 dataset)

📌 Key Features
🔍 1. CNN-Based Face Recognition
Uses FaceNet for generating 512-D embeddings

Dataset examples:

LFW (benchmark verification)

VGGFace2 (high-variance training)

CelebA (additional face images)

Custom images (dataset_family/)

Embedding comparison using Cosine Similarity

FAISS index for fast nearest-neighbor search

Detects:

Known Person → “Access Granted”

Unknown Person → Email Alert

🎥 2. Real-Time Surveillance
Live webcam feed

MTCNN detects face bounding boxes

Preprocessing + embedding extraction

Real-time inference using optimized CPU/GPU pipelines

📩 3. Automatic Email Alerts for Unknown Faces
When an unknown person appears:

✔ Captures face image
✔ Sends alert using Gmail SMTP
✔ Attaches detected person snapshot
✔ Logs the timestamp & identity confidence

🌐 4. Smart Web Interface (Node.js + Express)
Frontend Dashboard:

Live Monitoring Window

Alert History

Known Persons List

System Status

Backend:

Node.js (Express) for routing

Communicates with Flask API for recognition

🧠 5. Deep Learning Pipeline 
Face Detection
MTCNN

Landmark-based alignment

Cropping + resizing (160×160)

Face Embeddings
FaceNet model (pretrained)

Output: 512-D normalized vector

Similarity Matching
Cosine similarity

FAISS Index (optional)

Threshold-based classification

Emotion Recognition (Optional)
Trained on FER2013

Predicts 7 emotions:
angry, disgust, fear, happy, neutral, sad, surprise



🛠️ Technologies Used
🔧 Backend
Python Flask

Node.js + Express

REST APIs for communication

🤖 Machine Learning
TensorFlow/Keras

MTCNN

FaceNet

OpenCV

Scikit-learn (ROC/AUC)

📡 Communication
Gmail SMTP

Flask ↔ Node API integration

💻 Frontend
HTML

CSS

JavaScript

🚀 Installation & Setup
1️⃣ Clone Repository

git clone https://github.com/aashi-121/Smart-Surveillance-System.git
cd Smart-Surveillance-System
2️⃣ Install Python Dependencies
bash
Copy code
pip install -r requirements.txt
If TensorFlow errors:

bash
Copy code
pip install tensorflow==2.12.0
3️⃣ Install Node Dependencies

npm install
📥 Dataset Setup
LFW Dataset
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

VGGFace2 Dataset
https://www.kaggle.com/datasets/hearfool/vggface2

FER2013 Dataset (Emotion Detection)
https://www.kaggle.com/datasets/msambare/fer2013

Place datasets inside:


datasets/
🧬 Embedding Generation (FaceNet)
Step 1: Enroll Faces

python scripts/enroll.py
Step 2: Build FAISS Index

python scripts/build_index.py
🎥 Run Real-Time Surveillance

python scripts/realtime_demo.py
🙂 Run Real-Time Emotion Detection 


python scripts/emotion_realtime.py
📊 LFW Evaluation 


python scripts/evaluate_lfw.py
Outputs:

ROC curve

Verification accuracy

Best threshold

Similarity matrix

📧 Email Alert Configuration
In app.py, edit:

python

SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"
Generate app password for Gmail:
https://myaccount.google.com/apppasswords

🔮 Future Improvements
Integrate ArcFace (InsightFace)

Add door-lock control via IoT

Mobile app interface

Add motion detection

Add database for logging

✨ Authors
Aashi & Shreshtha Kushwaha
B.E. CSE (AIML)
Smart Home Surveillance Capstone Project
