import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialisation des compteurs et des états
exercise_counts = {"squats": 0, "situps": 0, "pushups": 0, "jumping_jacks": 0, "pullups": 0}
previous_state = None
current_exercise = None  # Exercice actif

# Fonction pour calculer l'angle entre trois points
def calculate_angle(a, b, c):
    angle = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]))
    if angle < 0:
        angle += 360
    return angle

# Fonction pour détecter l'état de l'exercice
def detect_exercise_state(landmarks, exercise):
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

    thresholds = {
        "squats": 90,   # Squats : angle au niveau du genou
        "pushups": 90,  # Push-ups : angle entre bras et tronc
        "jumping_jacks": 90,  # Jumping jacks : angle des bras
        "pullups": 45   # Pull-ups : angle entre épaules et coudes
    }

    if exercise == "squats":
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        return "down" if knee_angle < thresholds["squats"] else "up"

    elif exercise == "pushups":
        shoulder_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        return "down" if shoulder_angle > thresholds["pushups"] else "up"

    elif exercise == "jumping_jacks":
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        return "down" if left_arm_angle < thresholds["jumping_jacks"] else "up"

    elif exercise == "pullups":
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        return "up" if left_arm_angle < thresholds["pullups"] else "down"

    return "no_pose"

# Traitement vidéo
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Erreur : Impossible de lire la caméra.")
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Gestion des exercices
            if current_exercise is None:
                # Déterminer l'exercice actif
                for exercise in exercise_counts.keys():
                    state = detect_exercise_state(results.pose_landmarks.landmark, exercise)
                    if state != "no_pose":  # Commencer avec le premier mouvement valide
                        current_exercise = exercise
                        print(f"Exercice détecté : {current_exercise}")
                        break
            else:
                # Continuer avec l'exercice actif
                new_state = detect_exercise_state(results.pose_landmarks.landmark, current_exercise)
                print(f"Exercice : {current_exercise}, État actuel : {new_state}, État précédent : {previous_state}")

                # Comptabiliser un cycle "down" → "up"
                if new_state == "up" and previous_state == "down":
                    exercise_counts[current_exercise] += 1
                    print(f"{current_exercise} compté !")

                previous_state = new_state

                # Réinitialiser si aucun mouvement significatif n'est détecté
                if new_state == "no_pose":
                    current_exercise = None
                    previous_state = None

        # Afficher les résultats
        y_position = 30
        for ex, count in exercise_counts.items():
            cv2.putText(image, f"{ex.capitalize()}: {count}", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_position += 30

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
