import os
import sys
import io

# Force UTF-8 encoding for stdout to prevent UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Optionally, reduce TensorFlow logging (only show errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from keras.models import model_from_json
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def load_model(model_json_file, model_weights_file):
    """Load the Keras model from JSON and weights."""
    with open(model_json_file, "r", encoding="utf-8") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(model_weights_file)
    return model

# Load the emotion recognition model
model = load_model("emotionRecognition.json", "emotionRecognition.keras")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define emotion labels
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

def extract_features(image):
    """Prepare the face image for model prediction."""
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    # Decode image in grayscale mode for face detection
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        return jsonify({'faces': []})

    results = []
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        features = extract_features(face_img)
        pred = model.predict(features)
        emotion = labels[pred.argmax()]

        results.append({
            'emotion': emotion,
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h)
        })

    return jsonify({'faces': results})

if __name__ == '__main__':
    app.run(debug=True)
