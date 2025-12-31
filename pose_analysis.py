import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def analyze_pose(video_path):
    cap = cv2.VideoCapture(video_path)
    scores = []

    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image)

            if result.pose_landmarks:
                scores.append(1)

    cap.release()
    return min(100, len(scores))