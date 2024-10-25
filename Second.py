import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                # Extract landmark coordinates
                landmarks = face_landmarks.landmark
                # Example: Analyze the distance between certain landmarks to infer expression
                left_mouth = np.array([landmarks[61].x, landmarks[61].y])
                right_mouth = np.array([landmarks[291].x, landmarks[291].y])
                
                # Calculate distances
                #mouth_width = np.linalg.norm(left_mouth - right_mouth)
                mouth_width = abs(left_mouth[0] - right_mouth[0])

                #printing on the console just for testing
                print(mouth_width)

                # Simple expression analysis based on distances
                # If it does not work adjust the threshold values 0.1 and 0.08
                if mouth_width > 0.15:
                    expression = "Smiling"
                elif mouth_width < 0.08:
                    expression = "Kissing"
                else:
                    expression = "Neutral"

                # Display expression
                cv2.putText(image, expression, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the output
        cv2.imshow('Facial Expression Analysis', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()