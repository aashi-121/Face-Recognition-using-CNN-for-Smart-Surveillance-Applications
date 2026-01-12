// DOM Elements
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const result = document.getElementById("result");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const captureBtn = document.getElementById("captureBtn");

// Variables for webcam stream & analysis
let stream = null;
let analyzeInterval = null;
let analyzing = false;

// Start the webcam and begin analysis
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    analyzing = true;

    // Update button states and status text
    startBtn.disabled = true;
    stopBtn.disabled = false;
    captureBtn.disabled = false;
    result.textContent = "Camera started. Analyzing...";

    // Analyze frames every 500ms
    analyzeInterval = setInterval(analyzeFrame, 500);
  } catch (err) {
    console.error("Error accessing webcam:", err);
    result.textContent = "Error accessing webcam.";
  }
}

// Stop the webcam and analysis
function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  analyzing = false;
  clearInterval(analyzeInterval);
  video.srcObject = null;

  startBtn.disabled = false;
  stopBtn.disabled = true;
  captureBtn.disabled = true;
  result.textContent = "Camera stopped.";

  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Capture the current frame and send to API for analysis
function analyzeFrame() {
  if (!analyzing || !video.videoWidth || !video.videoHeight) return;

  // Set canvas dimensions to match the video feed
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  
  // Draw the video frame onto the canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert the canvas to a Blob and send it to the API
  canvas.toBlob(sendFrame, "image/jpeg");
}

// Send the captured frame to the backend API
function sendFrame(blob) {
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");

  // Update the URL to your deployed API endpoint when ready
  fetch("http://localhost:5000/analyze", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      // Redraw the current video frame on the canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      if (data.faces && data.faces.length > 0) {
        // Draw bounding boxes and labels for each detected face
        data.faces.forEach((face) => {
          ctx.strokeStyle = "red";
          ctx.lineWidth = 2;
          ctx.strokeRect(face.x, face.y, face.w, face.h);
          ctx.fillStyle = "red";
          ctx.font = "18px Arial";
          ctx.fillText(face.emotion, face.x, face.y - 5);
        });
        // Display the first detected emotion
        result.textContent = "Detected Emotion: " + data.faces[0].emotion;
      } else {
        result.textContent = "No face detected";
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      result.textContent = "Error analyzing frame.";
    });
}

// Capture the current frame (with annotations) and download it as an image
function captureFrame() {
  if (!video.videoWidth || !video.videoHeight) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataURL = canvas.toDataURL("image/png");
  const link = document.createElement("a");
  link.href = dataURL;
  link.download = "capture.png";
  link.click();
}

// Event listeners for button actions
document.addEventListener("DOMContentLoaded", () => {
  startBtn.addEventListener("click", startCamera);
  stopBtn.addEventListener("click", stopCamera);
  captureBtn.addEventListener("click", captureFrame);
});
