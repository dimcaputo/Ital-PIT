import cv2
import mediapipe as mp
import csv
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
import pickle
import tensorflow as tf

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculer_angle(A, B, C):
    # Calculer les vecteurs AB et BC
    AB = np.array(B) - np.array(A)
    BC = np.array(C) - np.array(B)

    # Calculer le produit scalaire de AB et BC
    dot_product = np.dot(AB, BC)

    # Calculer les normes des vecteurs AB et BC
    norm_AB = np.linalg.norm(AB)
    norm_BC = np.linalg.norm(BC)

    # Calculer l'angle en radians
    angle_rad = np.arccos(dot_product / (norm_AB * norm_BC))

    # Convertir l'angle en degrés
    angle_deg = np.degrees(angle_rad)

    return angle_deg

cap = cv2.VideoCapture(0)

landmark_names = [
    "Nose", "LeftEyeInner", "LeftEye", "LeftEyeOuter", "RightEyeInner", "RightEye", "RightEyeOuter",
    "LeftEar", "RightEar", "MouthLeft", "MouthRight", "LeftShoulder", "RightShoulder",
    "LeftElbow", "RightElbow", "LeftWrist", "RightWrist", "LeftPinky", "RightPinky",
    "LeftIndex", "RightIndex", "LeftThumb", "RightThumb", "LeftHip", "RightHip",
    "LeftKnee", "RightKnee", "LeftAnkle", "RightAnkle", "LeftHeel", "RightHeel",
    "LeftFootIndex", "RightFootIndex"
]
with tf.device('/CPU:0'):
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
                coordinates = []
                positions = results.pose_landmarks.landmark
                pose_data = np.zeros(shape=(3 * len(results.pose_landmarks.landmark) + 7,)) 
                j = 0
                for position in positions:
                    coordinates.append((position.x, position.y, position.z))
                    pose_data[j] = position.x
                    pose_data[j+1] = position.y
                    pose_data[j+2] = position.z
                    j = j + 3

                pose_data[3*len(results.pose_landmarks.landmark)+0] = calculer_angle(coordinates[14], coordinates[12], coordinates[24])                         # right_elbow_right_shoulder_right_hip
                pose_data[3*len(results.pose_landmarks.landmark)+1] = calculer_angle(coordinates[13], coordinates[11], coordinates[23])                         # left_elbow_left_shoulder_left_hip
                pose_data[3*len(results.pose_landmarks.landmark)+2] = calculer_angle(coordinates[26], tuple((x + y) / 2 for x, y in zip(coordinates[23], coordinates[24])), coordinates[25])  # right_knee_mid_hip_left_knee
                pose_data[3*len(results.pose_landmarks.landmark)+3] = calculer_angle(coordinates[24], coordinates[26], coordinates[28])                         # right_hip_right_knee_right_ankle
                pose_data[3*len(results.pose_landmarks.landmark)+4] = calculer_angle(coordinates[23], coordinates[25], coordinates[27])                         # left_hip_left_knee_left_ankle
                pose_data[3*len(results.pose_landmarks.landmark)+5] = calculer_angle(coordinates[16], coordinates[14], coordinates[12])                         # right_wrist_right_elbow_right_shoulder
                pose_data[3*len(results.pose_landmarks.landmark)+6] = calculer_angle(coordinates[15], coordinates[13], coordinates[11])    
                

                prediction = model.predict(pose_data.reshape(1, 3 * len(results.pose_landmarks.landmark) + 7), verbose=0)
                index_class = np.argmax(prediction, axis=1)
                class_predicted = classes[index_class]
                    
                print(class_predicted)

                # Dessiner les landmarks sur l'image
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
