import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from keras.models import load_model
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Charger le modèle et les classes
model = load_model('/home/selma.khounati@Digital-Grenoble.local/Documents/19. ACV/Ital-PIT-dimitri/classifier.keras')
classes = pd.read_csv('/home/selma.khounati@Digital-Grenoble.local/Documents/19. ACV/Ital-PIT-dimitri/classes.csv', names=[n for n in range(11)]).values[0]

cap = cv2.VideoCapture(0)

exercise_counts = {"squats": 0, "situp": 0, "jumping_jacks": 0, "pullups": 0, "pushups": 0}
current_exercise = None
previous_state = None

# Fonction de détection de l'état

def detect_exercise_state(landmarks, exercise):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y

    thresholds = {
        "squats": 0.08,
        "situp": 0.12,
        "jumping_jacks": 0.07,
        "pullups": 0.12,
        "pushups": 0.12
    }
    threshold = thresholds.get(exercise, 0.1)

    if exercise == "squats":
        return "down" if left_hip - left_knee > threshold else "up"
    elif exercise == "situp":
        return "down" if left_shoulder > left_hip + threshold else "up"
    elif exercise == "jumping_jacks":
        return "up" if left_wrist > left_shoulder + threshold else "down"
    elif exercise == "pullups":
        return "up" if left_elbow < left_shoulder - threshold else "down"
    elif exercise == "pushups":
        return "down" if left_shoulder > left_hip + threshold else "up"
    return "no_pose"

# Traitement vidéo
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Erreur : Impossible de lire la caméra.")
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            pose_data = np.array([val for landmark in results.pose_landmarks.landmark for val in (landmark.x, landmark.y, landmark.z)])
            # print("Pose data:", pose_data)

            prediction = model.predict(pose_data.reshape(1, 99))
            index_class = np.argmax(prediction)
            confidence = np.max(prediction)
            class_predicted = classes[int(index_class)]
            print(f"Exercice prédit: {class_predicted} (Confiance: {confidence:.2f})")

            new_state = detect_exercise_state(results.pose_landmarks.landmark, class_predicted)
            print(f"Nouvel état détecté : {new_state}")

            if new_state != "no_pose" and new_state != previous_state:
                time.sleep(0.2)  # Éviter le comptage excessif
                print(f"Exercice: {class_predicted} - Position: {new_state}")
                previous_state = new_state
                if new_state == "up":
                    exercise_counts[class_predicted] += 1
                    print(f"{class_predicted}: {exercise_counts[class_predicted]}")

        # Affichage écran
        text = "Exercices:\n" + "\n".join([f"{ex.capitalize()}: {count}" for ex, count in exercise_counts.items()])
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
