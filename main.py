import mediapipe as mp
import cv2
import time
import pyautogui
from collections import deque
from pynput.mouse import Controller as MouseController

# Prevent pyautogui from throwing errors at screen corners
pyautogui.FAILSAFE = False

# --- Setup hand landmarker ---
base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task")
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

# Hand connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
FINGERTIPS = {4, 8, 12, 16, 20}

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 128)

# --- Swipe tracking ---
SWIPE_HISTORY_SIZE = 15
SWIPE_MIN_DISTANCE = 0.15
SWIPE_TIME_WINDOW = 0.5
wrist_history = deque(maxlen=SWIPE_HISTORY_SIZE)
last_swipe = ""
last_swipe_time = 0
SWIPE_COOLDOWN = 0.8

# --- Scroll settings ---
SCROLL_AMOUNT = 500

# --- Mouse control settings ---
SCREEN_W, SCREEN_H = pyautogui.size()
MOUSE_SENSITIVITY = 1.5
DEAD_ZONE = 0.005          # ignore movements smaller than this to reduce jitter
ZOT_SMOOTH_FRAMES = 5      # number of frames to average position over

mouse = MouseController()
zot_prev_pos = None
zot_position_history = deque(maxlen=ZOT_SMOOTH_FRAMES)


def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def is_finger_extended(landmarks, finger, hand_label):
    fingers = {
        "thumb":  [4, 3, 2, 1],
        "index":  [8, 7, 6, 5],
        "middle": [12, 11, 10, 9],
        "ring":   [16, 15, 14, 13],
        "pinky":  [20, 19, 18, 17],
    }
    tip, dip, pip_, mcp = [landmarks[i] for i in fingers[finger]]

    if finger == "thumb":
        index_mcp = landmarks[5]
        if hand_label == "Right":
            return tip.x < index_mcp.x
        else:
            return tip.x > index_mcp.x
    else:
        return tip.y < pip_.y


def detect_gesture(landmarks, hand_label):
    """Detect the 4-finger gesture and the zot gesture."""
    extended = {
        "thumb":  is_finger_extended(landmarks, "thumb", hand_label),
        "index":  is_finger_extended(landmarks, "index", hand_label),
        "middle": is_finger_extended(landmarks, "middle", hand_label),
        "ring":   is_finger_extended(landmarks, "ring", hand_label),
        "pinky":  is_finger_extended(landmarks, "pinky", hand_label),
    }

    # Zot: index + pinky extended, thumb/middle/ring tips all close together
    if extended["index"] and extended["pinky"]:
        thumb_tip  = landmarks[4]
        middle_tip = landmarks[12]
        ring_tip   = landmarks[16]
        ZOT_THRESHOLD = 0.04
        if (dist(thumb_tip, middle_tip) < ZOT_THRESHOLD and
                dist(thumb_tip, ring_tip) < ZOT_THRESHOLD and
                dist(middle_tip, ring_tip) < ZOT_THRESHOLD):
            return "Zot", YELLOW

    # 4: index, middle, ring, pinky extended — thumb closed
    if (not extended["thumb"] and extended["index"] and extended["middle"]
            and extended["ring"] and extended["pinky"]):
        return "", WHITE
        # return "4", GREEN
    return "4", GREEN
    # return "", WHITE


def detect_swipe(landmarks, current_time):
    """Detect swipe gestures based on wrist movement over time."""
    global last_swipe, last_swipe_time

    wrist = landmarks[0]
    wrist_history.append((wrist.x, wrist.y, current_time))

    if len(wrist_history) < 5:
        return None

    if current_time - last_swipe_time < SWIPE_COOLDOWN:
        return None

    oldest = None
    for pt in wrist_history:
        if current_time - pt[2] <= SWIPE_TIME_WINDOW:
            oldest = pt
            break

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
        last_swipe = swipe
        last_swipe_time = current_time
        wrist_history.clear()

    return swipe


def handle_swipe_action(swipe):
    """Translate a swipe into a scroll command."""
    if swipe == "Swipe Up":
        pyautogui.scroll(SCROLL_AMOUNT)
    elif swipe == "Swipe Down":
        pyautogui.scroll(-SCROLL_AMOUNT)
    elif swipe == "Swipe Left":
        pyautogui.scroll(-SCROLL_AMOUNT)
    elif swipe == "Swipe Right":
        pyautogui.scroll(SCROLL_AMOUNT)


def handle_zot_mouse(landmarks):
    """Move the mouse relative to hand movement with smoothing and dead zone."""
    global zot_prev_pos

    thumb_tip  = landmarks[4]
    middle_tip = landmarks[12]
    ring_tip   = landmarks[16]

    avg_x = (thumb_tip.x + middle_tip.x + ring_tip.x) / 3
    avg_y = (thumb_tip.y + middle_tip.y + ring_tip.y) / 3

    # Add to rolling average history
    zot_position_history.append((avg_x, avg_y))

    # Smooth position over last N frames
    smoothed_x = sum(p[0] for p in zot_position_history) / len(zot_position_history)
    smoothed_y = sum(p[1] for p in zot_position_history) / len(zot_position_history)

    if zot_prev_pos is not None:
        dx = smoothed_x - zot_prev_pos[0]
        dy = smoothed_y - zot_prev_pos[1]

        # Only move if outside the dead zone
        if abs(dx) > DEAD_ZONE or abs(dy) > DEAD_ZONE:
            move_x = int(dx * SCREEN_W * MOUSE_SENSITIVITY)
            move_y = int(dy * SCREEN_H * MOUSE_SENSITIVITY)
            mouse.move(move_x, move_y)  # pynput is faster than pyautogui

    zot_prev_pos = (smoothed_x, smoothed_y)


def get_hand_scale(landmarks, w, h, reference_size=150):
    """Calculate a scale factor based on how large the hand appears on screen."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    dx = (wrist.x - middle_mcp.x) * w
    dy = (wrist.y - middle_mcp.y) * h
    hand_size = (dx**2 + dy**2) ** 0.5

    return max(0.3, min(2.0, hand_size / reference_size))


def get_bounding_box(landmarks, w, h, padding=20):
    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]
    return (
        max(0, int(min(x_coords)) - padding),
        max(0, int(min(y_coords)) - padding),
        min(w, int(max(x_coords)) + padding),
        min(h, int(max(y_coords)) + padding),
    )


def draw_hand(img, landmarks, handedness, w, h, gesture, gesture_color):
    x1, y1, x2, y2 = get_bounding_box(landmarks, w, h)
    cv2.rectangle(img, (x1, y1), (x2, y2), gesture_color, 2)

    hand_label = handedness[0].category_name
    text = f"{hand_label}: {gesture}" if gesture else hand_label
    cv2.putText(img, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)

    for c in HAND_CONNECTIONS:
        p1 = landmarks[c[0]]
        p2 = landmarks[c[1]]
        pt1 = (int(p1.x * w), int(p1.y * h))
        pt2 = (int(p2.x * w), int(p2.y * h))
        cv2.line(img, pt1, pt2, YELLOW, 2)

    scale = get_hand_scale(landmarks, w, h)

    for i, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        if i in FINGERTIPS:
            radius = max(1, int(8 * scale))
            cv2.circle(img, (cx, cy), radius, RED, -1)
            cv2.circle(img, (cx, cy), radius, WHITE, 2)
        else:
            radius = max(1, int(5 * scale))
            cv2.circle(img, (cx, cy), radius, BLUE, -1)


def draw_swipe_indicator(img, swipe, w, h):
    """Draw a big swipe arrow/text on screen."""
    if not swipe:
        return

    center_x, center_y = w // 2, h // 2
    arrow_len = 100

    if "Left" in swipe:
        cv2.arrowedLine(img, (center_x + arrow_len, center_y),
                        (center_x - arrow_len, center_y), ORANGE, 4, tipLength=0.3)
    elif "Right" in swipe:
        cv2.arrowedLine(img, (center_x - arrow_len, center_y),
                        (center_x + arrow_len, center_y), ORANGE, 4, tipLength=0.3)
    elif "Up" in swipe:
        cv2.arrowedLine(img, (center_x, center_y + arrow_len),
                        (center_x, center_y - arrow_len), ORANGE, 4, tipLength=0.3)
    elif "Down" in swipe:
        cv2.arrowedLine(img, (center_x, center_y - arrow_len),
                        (center_x, center_y + arrow_len), ORANGE, 4, tipLength=0.3)

    cv2.putText(img, swipe, (center_x - 80, center_y - arrow_len - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, ORANGE, 3)


# --- Webcam loop ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # @TODO unflip during demo
    h, w, _ = frame.shape
    small = cv2.resize(frame, (640, 360))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    now = time.time()

    result = detector.detect(mp_image)

    swipe = None
    if result.hand_landmarks:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            hand_label = handedness[0].category_name
            gesture, color = detect_gesture(landmarks, hand_label)
            draw_hand(frame, landmarks, handedness, w, h, gesture, color)

            if gesture == "4":
                swipe = detect_swipe(landmarks, now)
                zot_prev_pos = None
                zot_position_history.clear()
            elif gesture == "Zot":
                handle_zot_mouse(landmarks)
            else:
                wrist_history.clear()
                zot_prev_pos = None
                zot_position_history.clear()

    # If a swipe was detected, send the scroll command
    if swipe:
        handle_swipe_action(swipe)

    if swipe or (last_swipe and now - last_swipe_time < 0.6):
        draw_swipe_indicator(frame, swipe or last_swipe, w, h)

    num = len(result.hand_landmarks) if result.hand_landmarks else 0
    cv2.putText(frame, f"Hands: {num}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

    legend = [
        "4: index+middle+ring+pinky extended, thumb closed — swipe to scroll",
        "Zot: index+pinky extended, thumb+middle+ring pinched — moves cursor",
        "Swipe: Hold 4 and move hand L/R/U/D quickly",
    ]
    for i, line in enumerate(legend):
        cv2.putText(frame, line, (10, h - 60 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    cv2.putText(frame, "Press Q to quit", (w - 170, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()