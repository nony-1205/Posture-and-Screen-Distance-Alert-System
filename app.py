# python -m venv venv
# venv\Scripts\activate

import cv2
import numpy as np
import streamlit as st


import cv2
import mediapipe as mp
import math
import streamlit as st

# Initialize MediaPipe Pose tracking tools
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_distance(p1, p2):
    """Calculates the straight-line pixel distance between two points."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# --- STREAMLIT WEB INTERFACE ---
st.set_page_config(page_title="Digital Well-being Monitor", layout="wide")
st.title("🧘 AI Posture & Screen-Distance Alert System")
st.write("Grade 10 Digital Well-being Project")

# Sidebar Controls for Customization
st.sidebar.header("🔧 Calibration Settings")
dist_threshold = st.sidebar.slider("Unsafe Distance Threshold (Eye Gap)", 0.05, 0.20, 0.09, step=0.01)
slouch_threshold = st.sidebar.slider("Slouching Shoulder-Ear Limit", 0.05, 0.20, 0.12, step=0.01)

run_app = st.checkbox("▶️ Start Webcam Monitor")
FRAME_WINDOW = st.image([]) # Placeholder to stream video inside the webpage

# --- MAIN APP LOOP ---
video_capture = cv2.VideoCapture(0) # Open default webcam

while run_app and video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        st.write("Failed to access webcam.")
        break

    # Flip frame horizontally for a natural mirror view, convert color to RGB for MediaPipe
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image frame through MediaPipe's AI
    results = pose.process(rgb_frame)

    # Initialize status alert strings
    distance_status = "Good Distance"
    posture_status = "Good Posture"
    alert_color = (0, 255, 0) # Green by default

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Extract necessary body coordinates
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # 📐 MATH LAYER 1: Calculate Screen Distance (Pixel gap between eyes)
        eye_distance = calculate_distance(left_eye, right_eye)
        if eye_distance > dist_threshold:
            distance_status = "⚠️ TOO CLOSE TO SCREEN!"
            alert_color = (0, 0, 255) # Turn visual text Red

        # 📐 MATH LAYER 2: Calculate Posture (Horizontal gap between Nose and Shoulder)
        # If your nose drops significantly forward past your shoulder, you are slouching.
        nose_to_shoulder_gap = abs(nose.x - left_shoulder.x)
        if nose_to_shoulder_gap > slouch_threshold:
            posture_status = "⚠️ SLOUCHING DETECTED!"
            alert_color = (0, 0, 255)

        # Draw the skeleton grid overlay on the video frame
        mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display real-time telemetry text directly on the screen
        cv2.putText(rgb_frame, f"Distance State: {distance_status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)
        cv2.putText(rgb_frame, f"Posture State: {posture_status}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)

    # Push the processed frame live into the Streamlit Web dashboard
    FRAME_WINDOW.image(rgb_frame)

else:
    video_capture.release()
    st.write("Webcam Stopped.")
