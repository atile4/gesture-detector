import mediapipe as mp
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Setup hand landmarker ---
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

# Hand connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm
]

FINGERTIPS = {4, 8, 12, 16, 20}

# Colors (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


def get_bounding_box(landmarks, w, h, padding=20):
    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]
    return (
        max(0, int(min(x_coords)) - padding),
        max(0, int(min(y_coords)) - padding),
        min(w, int(max(x_coords)) + padding),
        min(h, int(max(y_coords)) + padding),
    )


def draw_hand(img, landmarks, handedness, w, h):
    x1, y1, x2, y2 = get_bounding_box(landmarks, w, h)
    cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)

    label = handedness[0].category_name
    conf = handedness[0].score
    cv2.putText(img, f"{label} ({conf:.0%})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

    # Draw connections
    for c in HAND_CONNECTIONS:
        p1 = landmarks[c[0]]
        p2 = landmarks[c[1]]
        pt1 = (int(p1.x * w), int(p1.y * h))
        pt2 = (int(p2.x * w), int(p2.y * h))
        cv2.line(img, pt1, pt2, YELLOW, 2)

    # Draw landmarks
    for i, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        if i in FINGERTIPS:
            cv2.circle(img, (cx, cy), 8, RED, -1)
            cv2.circle(img, (cx, cy), 8, WHITE, 2)
        else:
            cv2.circle(img, (cx, cy), 5, BLUE, -1)


# --- Webcam loop ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            draw_hand(frame, landmarks, handedness, w, h)

    num = len(result.hand_landmarks) if result.hand_landmarks else 0
    cv2.putText(frame, f"Hands detected: {num}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    cv2.putText(frame, "Press Q to quit", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()