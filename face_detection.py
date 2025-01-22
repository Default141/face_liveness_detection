import cv2
import mediapipe as mp



def detect_face():
    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    # Open webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB (Mediapipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks on the face
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

                # Detect specific points, e.g., eyes
                left_eye = face_landmarks.landmark[133]  # Example landmark
                right_eye = face_landmarks.landmark[362]

                # Convert to image coordinates
                ih, iw, _ = frame.shape  # Correctly unpack frame dimensions
                left_eye_coords = (int(left_eye.x * iw), int(left_eye.y * ih))
                right_eye_coords = (int(right_eye.x * iw), int(right_eye.y * ih))

                # Draw eye landmarks on the image
                cv2.circle(frame, left_eye_coords, 5, (0, 255, 0), -1)
                cv2.circle(frame, right_eye_coords, 5, (0, 255, 0), -1)

        cv2.imshow("Liveness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "finish running"
