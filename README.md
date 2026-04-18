# TempoVision – AI Conducting Recognition System 🎼🤖

## 📌 Overview

TempoVision is a computer vision and machine learning project that interprets musical conducting gestures in real time. Using a webcam, the system tracks hand movements to estimate tempo (BPM) and time signature, then evaluates user performance against expected patterns.

The project uses deep learning to create a model used for an interactive conducting experience — effectively turning your webcam into a virtual orchestra assistant.

## 🚀 Features:

🎥 Real-time hand tracking using MediaPipe
 🧠 Machine learning model for:
    -Time signature classification
    -Tempo (BPM) estimation
🎼 Interactive playback:
    -Matches conducting tempo to closest song
    -Evaluates user accuracy in real time
 📊 High performance:
~96% accuracy (time signature classification)
~6.34 BPM mean absolute error

## 🏗️ Project Structure:

-recording.py      # Collects gesture data using webcam
-train.py          # Preprocesses data & trains ML model
-live.py           # Runs real-time conducting detection
-dataset/          # Stored training data
-songs/            # Audio files for playback
-conductor_ai.h5   # Saved trained model

## ⚙️ How It Works:

### 1. Data Collection

Run:

python recording.py

-Records conducting gestures via webcam
-Uses MediaPipe to extract hand landmark coordinates over time
-Saves labeled time-series data into dataset/

### 2. Model Training

Run:

python train.py

-Interpolates and normalizes time-series data
-Trains a neural network using TensorFlow
-Outputs trained model for inference

### 3. Live Inference

Run:

python live.py

-Detects conducting gestures in real time
-Estimates BPM and time signature
-Selects closest matching song
-Evaluates user performance dynamically

## 🧰 Technologies Used
Python
TensorFlow / Keras
MediaPipe (Hand Landmark Detection)
OpenCV
NumPy / Pandas
SciPy
Scikit-learn
Pygame
