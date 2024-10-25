import cv2
import mediapipe as mp
import numpy as np


def calculate_au(landmarks):
    # Extract landmark coordinates in frame x
    left_eye_current = np.array([landmarks[33].x, landmarks[33].y])
    right_eye_current = np.array([landmarks[263].x, landmarks[263].y])
    left_mouth_current = np.array([landmarks[61].x, landmarks[61].y])
    right_mouth_current = np.array([landmarks[291].x, landmarks[291].y])

    # Compute eye distance for the current frame
    eye_distance_current = np.linalg.norm(left_eye_current - right_eye_current)

    # Compute mouth distance for the current frame
    mouth_width_current = np.linalg.norm(left_mouth_current - right_mouth_current)

    # Calculate the ratio for the current frame
    current_mouth_to_eye_ratio = mouth_width_current / eye_distance_current

    # Set a threshold for AU 12 (smile)
    threshold = 0.05  # Adjust this threshold as needed
    au12 = current_mouth_to_eye_ratio - neutral_mouth_to_eye_ratio > threshold

    return au12, current_mouth_to_eye_ratio


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize neutral landmarks and other variables
neutral_landmarks = None

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
    # Capture a neutral frame
    ret, neutral_frame = cap.read()
    if ret:
        neutral_image_rgb = cv2.cvtColor(neutral_frame, cv2.COLOR_BGR2RGB)
        neutral_results = face_mesh.process(neutral_image_rgb)
        if neutral_results.multi_face_landmarks:
            neutral_landmarks = neutral_results.multi_face_landmarks[0].landmark

            # Extract neutral landmark coordinates
            left_eye_frame1 = np.array([neutral_landmarks[33].x, neutral_landmarks[33].y])
            right_eye_frame1 = np.array([neutral_landmarks[263].x, neutral_landmarks[263].y])
            left_mouth_frame1 = np.array([neutral_landmarks[61].x, neutral_landmarks[61].y])
            right_mouth_frame1 = np.array([neutral_landmarks[291].x, neutral_landmarks[291].y])

            # distance between eyes in frame 1
            eye_distance_frame1 = np.linalg.norm(left_eye_frame1 - right_eye_frame1)

            # Avoid division by zero if the face is not detected (correclty)
            if eye_distance_frame1 == 0:
                exit

            # distance between mouth corners in frame 1
            mouth_width_frame1 = np.linalg.norm(left_mouth_frame1 - right_mouth_frame1)

            # normalized distance
            neutral_mouth_to_eye_ratio = mouth_width_frame1 / eye_distance_frame1

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks and neutral_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                # Calculate the presence of Action Unit 12 for the current frame
                au12, current_mouth_to_eye_ratio = calculate_au(face_landmarks.landmark)

                # Display results
                expression = f"AU 12: {'Smile' if au12 else 'Neutral'} | Current ratio: {current_mouth_to_eye_ratio:.2f} | Neutral Ratio: {neutral_mouth_to_eye_ratio:.2f}"
                cv2.putText(image, expression, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the output
        cv2.imshow('Facial Expression Analysis', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()