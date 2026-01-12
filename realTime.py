import sys
import argparse
import cv2
import numpy as np
from keras.models import model_from_json

# Force UTF-8 output (helps avoid Windows console encoding errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

def load_model(model_json_file, model_weights_file):
    """Load the Keras model from JSON and weights."""
    with open(model_json_file, "r", encoding="utf-8") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(model_weights_file)
    return model

def extract_features(image):
    """Reshape and scale the face image to match model input."""
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def main(args):
    # Load the emotion recognition model
    model = load_model(args.model_json, args.model_weights)
    
    # Load Haar Cascade for face detection
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)

    # Emotion labels
    labels = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'neutral',
        5: 'sad',
        6: 'surprise'
    }

    # Determine video source (webcam index or file path)
    if args.source.isdigit():
        video_source = int(args.source)
    else:
        video_source = args.source

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    window_name = "Real-Time Emotion Recognition"
    # Create a resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Optionally, set an initial window size (uncomment to use):
    # cv2.resizeWindow(window_name, 1280, 720)

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        try:
            for (x, y, w, h) in faces:
                # Extract the face region from grayscale
                face_img = gray[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Preprocess face for the model
                face_img = cv2.resize(face_img, (48, 48))
                features = extract_features(face_img)
                pred = model.predict(features)
                
                # Get emotion label
                label = labels[pred.argmax()]
                # Encode label to ASCII to avoid Unicode issues
                emotion_text = label.encode("ascii", "ignore").decode()
                
                cv2.putText(
                    frame,
                    emotion_text,
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    2,
                    (0, 0, 255),
                    2
                )
            
            # Show the frame in a resizable window
            cv2.imshow(window_name, frame)

            # Wait for 'interval' ms. Press 'q' to quit.
            key = cv2.waitKey(args.interval) & 0xFF
            if key == ord('q'):
                break
        except cv2.error as e:
            print("OpenCV error:", e)
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-Time Emotion Recognition using OpenCV and Keras")
    parser.add_argument(
        "--model_json",
        type=str,
        default="emotionRecognition.json",
        help="Path to the model JSON file (default: emotionRecognition.json)"
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default="emotionRecognition.keras",
        help="Path to the model weights file (default: emotionRecognition.keras)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (default: '0' for webcam, or provide a video file path)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=27,
        help="Frame capture interval in milliseconds (default: 27)."
    )
    args = parser.parse_args()
    main(args)
