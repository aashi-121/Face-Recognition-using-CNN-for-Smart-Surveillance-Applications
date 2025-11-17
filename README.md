# 🛡️ Smart Surveillance System using Face Recognition (CNN)

A Smart Home Security & Surveillance System that uses Convolutional Neural Networks (CNN) for face recognition. The system detects and verifies visitors using a pre-trained CNN model. When an unknown visitor is detected, the system sends an email alert with the captured image.

This project combines:

Python (Flask) for face recognition & email notifications

Node.js for backend routing

OpenCV for real-time camera processing

CNN-based face recognition model

Web UI using HTML templates

📌 Features
🔍 Face Recognition (CNN-Based)

Uses CNN embeddings to detect known vs unknown faces.

Dataset stored inside dataset_family/.

🎥 Real-Time Surveillance

Captures frames using webcam / external camera.

Recognizes faces in real time.

📩 Automatic Email Alert

When an unknown face appears, system triggers:

Image capture

Email alert via Gmail SMTP

Timestamped logs

🌐 Web Interface

Simple dashboard built using:

Node.js

Express

HTML templates

🧠 Dual Application Architecture

app.py → Handles HTTP server for recognition, API endpoints

face_recognition.py → CNN model detection logic

index.js → Node server for UI routes and API requests

`📂 Project Structure
Home-security-system/
│
├── dataset_family/            # Training images for family members
│   ├── abhishek.jpeg
│   ├── amisha.jpeg
│   ├── ...
│
├── templates/                 # Frontend HTML templates
│
├── handlers/                  # Node route handlers
├── models/                    # Node models (if any)
├── public/                    # CSS, JS, static assets
├── routes/                    # Node routes
│
├── index.js                   # Node backend
├── app.py                     # Python Flask server
├── face_recognition.py        # CNN-based face recognition logic
│
├── mailme.sh                  # Shell script for email config (if used)
│
├── package.json               # Node dependencies
├── package-lock.json
├── requirements.txt           # Python dependencies
└── .gitignore`
`

🛠️ Technologies Used
🔧 Backend

Python 3.x

Flask

Node.js

Express.js

🤖 Machine Learning

TensorFlow / Keras

CNN for feature extraction

OpenCV

NumPy

📡 Communication

SMTP (Email Service)

API Integration between Flask & Node

💻 Frontend

HTML5

CSS

JavaScript

⚙️ Installation & Setup
1️⃣ Clone the Repository
https://github.com/aashi-121/Face-Recognition-using-CNN-for-Smart-Surveillance-Applications.git
cd Home-security-system

2️⃣ Install Python Dependencies
pip install -r requirements.txt

3️⃣ Install Node Dependencies
npm install

4️⃣ Configure Email Alerts

Edit your email credentials in app.py:

SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"


Make sure to generate a Google App Password if using Gmail.

5️⃣ Run Python Recognition Server
python app.py

6️⃣ Run Node Backend
node index.js

🚀 Usage Flow

Start the Python & Node servers

Open the web interface:

http://localhost:3000


Live camera feed starts

CNN model detects face

If face is known → access granted

If unknown → email alert sent with captured image

🧪 Dataset Details

Your dataset is stored in:

/dataset_family/


Each image is used to generate embeddings for CNN.

Example:

abhishek.jpeg  
amisha.jpeg  
harshit.jpeg  
...


Add more images for higher accuracy.

📧 Email Alert Example

When an unknown face is detected, the system sends:

Subject: "⚠️ Alert! Unknown person detected"

Body: Includes time + message

Attachment: Captured intruder image

🤝 Contribution

Feel free to submit:

Pull Requests

Bug Fixes

Feature Enhancements

📜 License

This project is licensed under the MIT License.

✨ Author

Aaash and Shreshtha Kushwaha
B.E. CSE (AIML)
Smart Home Security Capstone Project
