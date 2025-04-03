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

model = load_model('cnn.keras')
classes = pd.read_csv('classes.csv', names=[n for n in range(4)]).values[0]

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Erreur : Impossible de lire la caméra.")
            break

        # Conversion en RGB pour MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (128,128), interpolation= cv2.INTER_LINEAR)
        image_resized = image_resized[np.newaxis,:,:,:]

        prediction = model.predict(image_resized)
        index_class = np.argmax(prediction, axis=1)
        class_predicted = classes[index_class]

        print(class_predicted)

        # Dessiner les landmarks sur l'image
       # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
