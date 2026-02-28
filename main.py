import mediapipe as mp
import cv2
import time
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_analyzer import HandAnalyzer, FINGERTIPS
from colors import *

analyzer = HandAnalyzer()

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

# --- Swipe tracking ---
SWIPE_HISTORY_SIZE = 15
SWIPE_MIN_DISTANCE = 0.15
SWIPE_TIME_WINDOW  = 0.5
SWIPE_COOLDOWN     = 0.8
wrist_history  = deque(maxlen=SWIPE_HISTORY_SIZE)
last_swipe      = ""
last_swipe_time = 0


def detect_swipe(landmarks, current_time):
    global last_swipe, last_swipe_time
    wrist = landmarks[0]
    wrist_history.append((wrist.x, wrist.y, current_time))
    if len(wrist_history) < 5:
        return None
    if current_time - last_swipe_time < SWIPE_COOLDOWN:
        return None
    oldest = next((pt for pt in wrist_history
                   if current_time - pt[2] <= SWIPE_TIME_WINDOW), None)
    if oldest is None:
        return None
    dx = wrist.x - oldest[0]
    dy = wrist.y - oldest[1]
    swipe = None
    if abs(dx) > SWIPE_MIN_DISTANCE and abs(dx) > abs(dy) * 1.5:
        swipe = "Swipe Right" if dx > 0 else "Swipe Left"
    elif abs(dy) > SWIPE_MIN_DISTANCE and abs(dy) > abs(dx) * 1.5:
        swipe = "Swipe Down" if dy > 0 else "Swipe Up"
    if swipe:
        last_swipe, last_swipe_time = swipe, current_time
        wrist_history.clear()
    return swipe


def draw_hand(img, landmarks, handedness, w, h, gesture, gesture_color):
    x1, y1, x2, y2 = analyzer.calc_bounding_box(landmarks, w, h)
    cv2.rectangle(img, (x1, y1), (x2, y2), gesture_color, 2)
    hand_label = handedness[0].category_name
    cv2.putText(img, f"{hand_label}: {gesture}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
    ]
    for c in HAND_CONNECTIONS:
        pt1 = (int(landmarks[c[0]].x * w), int(landmarks[c[0]].y * h))
        pt2 = (int(landmarks[c[1]].x * w), int(landmarks[c[1]].y * h))
        cv2.line(img, pt1, pt2, YELLOW, 2)
    for i, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        if i in FINGERTIPS:
            cv2.circle(img, (cx, cy), 8, RED, -1)
            cv2.circle(img, (cx, cy), 8, WHITE, 2)
        else:
            cv2.circle(img, (cx, cy), 5, BLUE, -1)


def draw_swipe_indicator(img, swipe, w, h):
    if not swipe:
        return
    cx, cy, L = w // 2, h // 2, 100
    arrows = {
        "Left":  ((cx + L, cy), (cx - L, cy)),
        "Right": ((cx - L, cy), (cx + L, cy)),
        "Up":    ((cx, cy + L), (cx, cy - L)),
        "Down":  ((cx, cy - L), (cx, cy + L)),
    }
    for direction, (p1, p2) in arrows.items():
        if direction in swipe:
            cv2.arrowedLine(img, p1, p2, ORANGE, 4, tipLength=0.3)
    cv2.putText(img, swipe, (cx - 80, cy - L - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, ORANGE, 3)


# --- Main loop ---
def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.time()
        result = detector.detect(mp_image)

        swipe = None

        hand_analyzer = HandAnalyzer()
        if result.hand_landmarks:
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                hand_analyzer.update_state(landmarks, w, h)

                hand_state = hand_analyzer.get_state()
                print(hand_state)

                draw_hand(frame, landmarks, handedness, w, h, hand_state.gesture, hand_state.color)
                swipe    = detect_swipe(landmarks, now)

        if swipe or (last_swipe and now - last_swipe_time < 0.6):
            draw_swipe_indicator(frame, swipe or last_swipe, w, h)

        num = len(result.hand_landmarks) if result.hand_landmarks else 0
        cv2.putText(frame, f"Hands: {num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

        for i, line in enumerate([
            "Gestures: Peace | Thumbs Up/Down | Fist",
            "Open Palm | Pointing | Rock On | Three",
            "Swipe: Move hand L/R/U/D quickly",
        ]):
            cv2.putText(frame, line, (10, h - 60 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        cv2.putText(frame, "Press Q to quit", (w - 170, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        cv2.imshow("Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()