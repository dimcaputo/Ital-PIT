import cv2
import mediapipe as mp
import csv
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
import pickle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

landmark_names = [
    "Nose", "LeftEyeInner", "LeftEye", "LeftEyeOuter", "RightEyeInner", "RightEye", "RightEyeOuter",
    "LeftEar", "RightEar", "MouthLeft", "MouthRight", "LeftShoulder", "RightShoulder",
    "LeftElbow", "RightElbow", "LeftWrist", "RightWrist", "LeftPinky", "RightPinky",
    "LeftIndex", "RightIndex", "LeftThumb", "RightThumb", "LeftHip", "RightHip",
    "LeftKnee", "RightKnee", "LeftAnkle", "RightAnkle", "LeftHeel", "RightHeel",
    "LeftFootIndex", "RightFootIndex"
]

model = load_model('classifier.keras')
classes = pd.read_csv('classes.csv', names=[n for n in range(11)]).values[0]

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Erreur : Impossible de lire la caméra.")
            break

        # Conversion en RGB pour MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        
        if results.pose_landmarks:
            pose_data = np.zeros(shape=3 * len(results.pose_landmarks.landmark))  

            # Ajouter chaque landmark X, Y, Z
            j = 0
            for landmark in results.pose_landmarks.landmark:
                    pose_data[j] = landmark.x
                    pose_data[j+1] = landmark.y
                    pose_data[j+2] = landmark.z
                    j = j + 3
            
            index_class = np.argmax(model.predict(pose_data.reshape(1,99), verbose=0), axis=1)
            class_predicted = classes[index_class]

            print(class_predicted)

            # Dessiner les landmarks sur l'image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
