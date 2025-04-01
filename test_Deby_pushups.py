import cv2
import mediapipe as md 

md_drawing = md.solutions.drawing_utils
md_drawing_styles = md.solutions.drawing_styles
md_pose = md.solutions.pose

def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

pushUpCount = 0
pushUpStart = 0
position = None

cap = cv2.VideoCapture(0)

with md_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print('empty camera')
            continue
        
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        result = pose.process(image)

        image_height, image_width, _ = image.shape

        # Vérification si les landmarks sont détectés
        if result.pose_landmarks:
            # Points spécifiques
            nosePoint = (int(result.pose_landmarks.landmark[0].x * image_width), int(result.pose_landmarks.landmark[0].y * image_height))
            leftWrist = (int(result.pose_landmarks.landmark[15].x * image_width), int(result.pose_landmarks.landmark[15].y * image_height))
            rightWrist = (int(result.pose_landmarks.landmark[16].x * image_width), int(result.pose_landmarks.landmark[16].y * image_height))
            leftShoulder = (int(result.pose_landmarks.landmark[11].x * image_width), int(result.pose_landmarks.landmark[11].y * image_height))
            rightShoulder = (int(result.pose_landmarks.landmark[12].x * image_width), int(result.pose_landmarks.landmark[12].y * image_height))

            # Calcul de la distance entre l'épaule droite et le poignet droit pour détecter si on commence ou non
            if distanceCalculate(rightShoulder, rightWrist) < 130:
                pushUpStart = 1
            elif pushUpStart and distanceCalculate(rightShoulder, rightWrist) > 250:
                pushUpCount += 1
                pushUpStart = 0  # Réinitialiser pour le prochain push-up

            # Afficher le nombre de push-ups
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 100)  # Position du texte
            fontScale = 2
            color = (255, 0, 0)
            thickness = 3
            image = cv2.putText(image, "Push-up count:  " + str(pushUpCount), org, font, fontScale, color, thickness, cv2.LINE_AA)

            # Liste pour les coordonnées de tous les points
            imlist = []

            md_drawing.draw_landmarks(image, result.pose_landmarks, md_pose.POSE_CONNECTIONS)

            # Remplir imlist avec les coordonnées des landmarks
            for id, im in enumerate(result.pose_landmarks.landmark):
                X, Y = int(im.x * image_width), int(im.y * image_height)
                imlist.append([id, X, Y])

            # Vérification des positions "down" et "up" pour les push-ups
            if len(imlist) != 0:
                # Position "down" : les épaules et les poignets sont plus proches de la caméra (z plus petit)
                if imlist[11][2] >= imlist[12][2] and imlist[13][2] >= imlist[14][2]:
                    position = "down"
                
                # Position "up" : les poignets et les épaules sont plus éloignés de la caméra (z plus grand)
                if imlist[11][2] <= imlist[12][2] and imlist[13][2] <= imlist[14][2] and position == "down":
                    position = "up"
                    pushUpCount += 1
                    print(f"Push-up Count: {pushUpCount}")

        # # Affichage de l'image avec les landmarks
        # image= cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow("Push-up counter",image)
        
        # Sortir de la boucle si la touche 'q' est pressée
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

