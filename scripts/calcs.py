# %%
import cv2
import torch
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from collections import deque

# %%
print(f"OpenCV: {cv2.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MediaPipe: {mp.__version__}")
print(f"NumPy: {np.__version__}")

# %%
# cap = cv2.VideoCapture(0)
# print("Press 'q' to quit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = face_mesh.process(rgb_frame)
    
#     # Draw landmarks if face detected
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             mp_drawing.draw_landmarks(
#                 image=frame,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
#             )
        
#         # Display status
#         cv2.putText(frame, "Face Detected", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     else:
#         cv2.putText(frame, "No Face", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
#     cv2.imshow('MediaPipe FaceMesh', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# face_mesh.close()

# %%
# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# 3D model points for head pose
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

# Thresholds
EAR_THRESHOLD = 0.25
YAW_THRESHOLD = 32

# %%
def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calculate_ear(eye_landmarks):
    vertical1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
    vertical2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
    horizontal = calculate_distance(eye_landmarks[0], eye_landmarks[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def get_eye_landmarks(face_landmarks, eye_indices):
    return [face_landmarks.landmark[i] for i in eye_indices]

# %%
def get_2d_points(face_landmarks, img_width, img_height):
    indices = [1, 152, 33, 263, 61, 291]
    points_2d = []
    for idx in indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        points_2d.append([x, y])
    return np.array(points_2d, dtype=np.float64)

def get_head_pose(face_landmarks, img_width, img_height):
    image_points = get_2d_points(face_landmarks, img_width, img_height)
    
    focal_length = img_width
    center = (img_width / 2, img_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
    
    pitch = euler_angles[0][0]
    yaw = euler_angles[1][0]
    roll = euler_angles[2][0]
    
    return pitch, yaw, roll

# %%
class FocusScorer:
    def __init__(self):
        # Thresholds
        self.EAR_THRESHOLD = 0.21
        self.EYE_CLOSED_FRAMES_SHORT = 15  
        self.EYE_CLOSED_FRAMES_LONG = 45   
        
        self.PHONE_FRAMES_SHORT = 5
        self.PHONE_FRAMES_LONG = 15
        
        self.eye_counter = 0
        self.phone_counter = 0
        
        self.prev_score = 100
        self.alpha = 0.2  # smoothing factor

    def update(self, ear, yaw, gaze_direction, phone_detected):
        """
        ear: float (eye aspect ratio)
        yaw: float (head yaw angle in degrees)
        gaze_direction: string ("center", "slight", "away")
        phone_detected: bool
        """

        if ear < self.EAR_THRESHOLD:
            self.eye_counter += 1
        else:
            self.eye_counter = 0

        if self.eye_counter > self.EYE_CLOSED_FRAMES_LONG:
            eye_penalty = 30
        elif self.eye_counter > self.EYE_CLOSED_FRAMES_SHORT:
            eye_penalty = 15
        else:
            eye_penalty = 0

        
        if gaze_direction == "center":
            gaze_penalty = 0
        elif gaze_direction == "slight":
            gaze_penalty = 10
        else:  # "away"
            gaze_penalty = 20
        if abs(yaw) < 10:
            head_penalty = 0
        elif abs(yaw) < 20:
            head_penalty = 10
        else:
            head_penalty = 20

        
        if phone_detected:
            self.phone_counter += 1
        else:
            self.phone_counter = 0

        if self.phone_counter > self.PHONE_FRAMES_LONG:
            phone_penalty = 40
        elif self.phone_counter > self.PHONE_FRAMES_SHORT:
            phone_penalty = 20
        else:
            phone_penalty = 0


        raw_score = 100 - (
            eye_penalty +
            gaze_penalty +
            head_penalty +
            phone_penalty
        )

        raw_score = max(0, min(100, raw_score))


        smooth_score = (
            self.alpha * raw_score +
            (1 - self.alpha) * self.prev_score
        )

        self.prev_score = smooth_score
        return round(smooth_score, 2)

# %%
# Initialize YOLO
model = YOLO('yolo12m.pt')

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("Models initialized!")

# %%
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

focus_scorer = FocusScorer()

low_focus_counter = 0
LOW_FOCUS_THRESHOLD = 50
LOW_FOCUS_FRAMES = 30   # ~1 sec at 30 FPS

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_height, img_width = frame.shape[:2]

    yolo_results = model(frame, verbose=False)
    phone_detected = False

    for box in yolo_results[0].boxes:
        if int(box.cls[0]) == 67 and float(box.conf[0]) > 0.4:
            phone_detected = True
            break

    annotated_frame = yolo_results[0].plot()


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    face_detected = face_results.multi_face_landmarks is not None

    avg_ear = 0
    yaw = 0
    gaze_direction = "center"

    if face_detected:
        face_landmarks = face_results.multi_face_landmarks[0]

        mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        # EAR
        left_eye = get_eye_landmarks(face_landmarks, LEFT_EYE)
        right_eye = get_eye_landmarks(face_landmarks, RIGHT_EYE)

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Head Pose
        pitch, yaw, roll = get_head_pose(face_landmarks, img_width, img_height)

        # Gaze Category
        if abs(yaw) < 10:
            gaze_direction = "center"
        elif abs(yaw) < 20:
            gaze_direction = "slight"
        else:
            gaze_direction = "away"

        # Display EAR & Yaw
        cv2.putText(annotated_frame, f"EAR: {avg_ear:.3f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(annotated_frame, f"Yaw: {yaw:.1f}", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


    # FOCUS SCORE COMPUTATION
    if face_detected:
        focus_score = focus_scorer.update(
            ear=avg_ear,
            yaw=yaw,
            gaze_direction=gaze_direction,
            phone_detected=phone_detected
        )
    else:
        focus_score = focus_scorer.update(
            ear=0,
            yaw=30,
            gaze_direction="away",
            phone_detected=phone_detected
        )


    focus_score = 0 if focus_score is None else focus_score

    if focus_score < LOW_FOCUS_THRESHOLD:
        low_focus_counter += 1
    else:
        low_focus_counter = 0

    show_alert = low_focus_counter > LOW_FOCUS_FRAMES

    cv2.putText(annotated_frame, f"Focus Score: {focus_score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    bar_width = int((focus_score / 100) * 200)

    cv2.rectangle(annotated_frame, (10, 50), (210, 75), (50,50,50), -1)

    if focus_score >= 70:
        bar_color = (0, 255, 0)       # Green
    elif focus_score >= 40:
        bar_color = (0, 255, 255)     # Yellow
    else:
        bar_color = (0, 0, 255)       # Red

    cv2.rectangle(annotated_frame, (10, 50),
                  (10 + bar_width, 75), bar_color, -1)

    if show_alert:

        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0),
                      (img_width, img_height), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.15,
                        annotated_frame, 0.85, 0, annotated_frame)

        cv2.putText(
            annotated_frame,
            "GET BACK TO WORK!",
            (img_width // 4, img_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    phone_text = "PHONE: YES" if phone_detected else "PHONE: NO"
    phone_color = (0,0,255) if phone_detected else (0,255,0)

    cv2.putText(annotated_frame, phone_text, (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, phone_color, 2)

    face_text = "FACE: YES" if face_detected else "FACE: NO"
    face_color = (0,255,0) if face_detected else (0,0,255)

    cv2.putText(annotated_frame, face_text, (10, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

    cv2.imshow('FocusAI - Attention Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()

# %%



