import cv2
import mediapipe as mp
import csv

mp_holistic = mp.solutions.holistic
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

# Ouvrir un fichier CSV pour sauvegarder les résultats
with open("pose_landmarks_per_pose.csv", "w", newline="") as file:
    writer = csv.writer(file)
    header = ["Frame"]
    for name in landmark_names:
        header.extend([f"{name}_X", f"{name}_Y", f"{name}_Z"])
    writer.writerow(header)

frame_count = 0  

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
        open("pose_landmarks_per_pose.csv", "a", newline="") as file:
    writer = csv.writer(file)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Erreur : Impossible de lire la caméra.")
            break

        # Conversion en RGB pour MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        
        if results.pose_landmarks:
            pose_data = [frame_count]  

            # Ajouter chaque landmark X, Y, Z
            for landmark in results.pose_landmarks.landmark:
                pose_data.extend([landmark.x, landmark.y, landmark.z])

            writer.writerow(pose_data)

            # Dessiner les landmarks sur l'image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        
        cv2.imshow('MediaPipe Pose', image)
        frame_count += 1
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
