🛡️ Real-Time Face Recognition for Smart Surveillance
A lightweight smart surveillance system using CNN-based facial recognition, OpenCV, and automated email alerts.

This project detects faces in real time using a CNN model and sends an email alert when an unknown face is detected. It is designed as a simple but effective home/office surveillance solution.

📌 Features
🔍 Face Recognition (CNN Based)
Uses a pre-trained CNN model

Extracts facial features using embeddings

Matches live faces with known dataset images

Recognizes:

Known user → Access granted

Unknown user → Sends alert email

📩 Email Alerts
When an unknown person is detected:

Captures the frame

Sends an email through Gmail SMTP

Includes the intruder's image

🎥 Real-Time Processing
Uses OpenCV to capture webcam frames

Face detection using Haar Cascade

Continuous monitoring

🌐 Flask Backend API
app.py runs an HTTP server

API endpoint triggers recognition

📁 Project Structure
graphql
Copy code
Smart-Surveillance/
│── app.py                 # Flask server
│── face_recognition.py    # CNN model logic (feature extraction + matching)
│── mailme.sh              # Optional email configuration script
│── requirements.txt       # Python dependencies
│── LICENSE
│── README.md
│── .gitignore
⚙️ Installation & Setup
1️⃣ Clone the repository

git clone https://github.com/aashi-121/Face-Recognition-using-CNN-for-Smart-Surveillance-Applications.git
cd Face-Recognition-using-CNN-for-Smart-Surveillance-Applications
2️⃣ Install dependencies

pip install -r requirements.txt
If TensorFlow causes errors, use:


pip install tensorflow==2.12.0
⚠️ If this still fails, install CPU version:


pip install tensorflow-cpu
3️⃣ Add images of known users
Create a folder named dataset_family/ (if not already created):


dataset_family/
    aashi.jpeg
    shreshtha.jpeg
    mom.jpeg
    dad.jpeg
Each image represents a known family member.

4️⃣ Configure Email Alerts
Open app.py and edit:

python
Copy code
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"
Generate an app password from Google:
👉 https://myaccount.google.com/apppasswords

🚀 Running the System
Start the Flask server:

python app.py
This will:

✔ Start recognition
✔ Access the webcam
✔ Identify faces
✔ Send alert emails for unknown users

📚 How It Works (High-Level)
1. Face Detection
Uses OpenCV Haar Cascade

Extracts face region from each webcam frame

2. Feature Extraction
In face_recognition.py:

Grayscale → Resize → Normalize → CNN Input

CNN model generates a feature vector (embedding)

3. Face Matching
Compares embedding with dataset embeddings

If similarity < threshold → Unknown

If Unknown → Email Alert

4. Email Alert
app.py handles:

Image capture

Encoding

SMTP sending

🧪 Testing
To test quickly:

Show your trained face → should display “Known”

Show a new face → Should trigger email alert

🔮 Future Enhancements
Replace Haar Cascade with MTCNN

Replace CNN model with FaceNet / ArcFace

Add logging dashboard

Add OTP-based door unlock system

Deploy using Docker

✨ Authors
Aashi & Shreshtha
B.E. CSE (AIML)

