import cv2
import mediapipe as mp
import numpy as np
import os

# Désactivation des optimisations OneDNN pour éviter certains avertissements
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Compteurs et états des exercices
exercise_counts = {"squats": 0, "situps": 0, "pushups": 0, "jumping_jacks": 0}
previous_states = {exercise: None for exercise in exercise_counts.keys()}

# Fonction pour calculer l'angle entre trois points
def calculate_angle(a, b, c):
    angle = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]))
    return angle + 360 if angle < 0 else angle

# Détection de l'état de l'exercice
def detect_exercise_state(landmarks, exercise):
    keypoints = {
        "left_hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
        "left_knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
        "left_ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
        "left_shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
        "left_elbow": [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
        "left_wrist": [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y],
        "right_shoulder": [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
        "right_elbow": [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
        "right_wrist": [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
    }

    thresholds = {"squats": 70, "situps": 30, "pushups": 100, "jumping_jacks": 110}

    if exercise == "squats":
        knee_angle = calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"])
        return "down" if knee_angle < thresholds["squats"] else "up"
    elif exercise == "situps":
        hip_angle = calculate_angle(keypoints["left_knee"], keypoints["left_hip"], keypoints["left_shoulder"])
        return "down" if hip_angle < thresholds["situps"] else "up"
    elif exercise == "pushups":
        shoulder_angle = calculate_angle(keypoints["left_shoulder"], keypoints["left_elbow"], keypoints["left_wrist"])
        return "down" if shoulder_angle > thresholds["pushups"] else "up"
    elif exercise == "jumping_jacks":
        left_arm_angle = calculate_angle(keypoints["left_shoulder"], keypoints["left_elbow"], keypoints["left_wrist"])
        right_arm_angle = calculate_angle(keypoints["right_shoulder"], keypoints["right_elbow"], keypoints["right_wrist"])
        return "down" if left_arm_angle < thresholds["jumping_jacks"] and right_arm_angle < thresholds["jumping_jacks"] else "up"
    
    return "NO POSE"

cap = cv2.VideoCapture(0)
cv2.namedWindow('Détection des Exercices', cv2.WINDOW_NORMAL)

# Détection avec MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Erreur : Impossible de lire la caméra.")
            break

        # Prétraitement de l'image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        exercise_detected = False

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            for exercise in exercise_counts.keys():
                new_state = detect_exercise_state(results.pose_landmarks.landmark, exercise)
                if new_state == "up" and previous_states[exercise] == "down":
                    exercise_counts[exercise] += 1
                    print(f"{exercise.capitalize()} compté : {exercise_counts[exercise]}")
                if new_state != "NO POSE":
                    exercise_detected = True
                previous_states[exercise] = new_state

        # Affichage "NO POSE" si aucun exercice détecté
        if not exercise_detected:
            cv2.putText(image, "NO POSE DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Affichage des résultats
        for i, (ex, count) in enumerate(exercise_counts.items()):
            cv2.putText(image, f"{ex.capitalize()}: {count}", (10, 80 + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Détection des Exercices', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

