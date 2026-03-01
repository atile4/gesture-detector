import multiprocessing
import ctypes
import numpy as np
import cv2
import time
import pyautogui
from collections import deque
from types import SimpleNamespace
from queue import Empty

from hand_analyzer import HandAnalyzer, FINGERTIPS
from colors import *

# Prevent pyautogui from throwing errors at screen corners
pyautogui.FAILSAFE = False

# --- Constants ---
INFER_W, INFER_H = 640, 360
FRAME_PIXELS = INFER_W * INFER_H * 3

SCREEN_W, SCREEN_H = pyautogui.size()
MOUSE_SENSITIVITY = 1.5

SCROLL_AMOUNT = 500

SWIPE_HISTORY_SIZE = 15
SWIPE_MIN_DISTANCE = 0.15
SWIPE_TIME_WINDOW  = 0.5
SWIPE_COOLDOWN     = 0.8

# Module-level analyzer used by draw_hand for bounding box
analyzer = HandAnalyzer()


# ---------------------------------------------------------------------------
# Inference child process
# ---------------------------------------------------------------------------

def inference_process(shm_frame, frame_counter, stop_flag, result_queue):
    """Runs MediaPipe detection and HandAnalyzer in a dedicated process."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from hand_analyzer import HandAnalyzer
    import numpy as np
    import time

    base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    hand_analyzer = HandAnalyzer()

    last_counter = 0

    while not stop_flag.value:
        current = frame_counter.value
        if current == last_counter:
            time.sleep(0.001)
            continue

        # Copy the latest frame out of shared memory
        with shm_frame.get_lock():
            frame_data = np.frombuffer(shm_frame.get_obj(), dtype=np.uint8).copy()
        last_counter = current

        frame_rgb = frame_data.reshape(INFER_H, INFER_W, 3)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        det_result = detector.detect(mp_image)

        hands = []
        if det_result.hand_landmarks:
            for landmarks, handedness in zip(det_result.hand_landmarks, det_result.handedness):
                hand_analyzer.update_state(landmarks, INFER_W, INFER_H)
                state = hand_analyzer.get_state()
                hands.append({
                    'landmarks_data': [(lm.x, lm.y, lm.z) for lm in landmarks],
                    'handedness_name': handedness[0].category_name,
                    'gesture': state.gesture,
                    'color': state.color,
                    'openness': state.openness,
                    'distance_from_cam': state.distance_from_cam,
                    'openness_history': list(state.openness_history),
                })

        # Drop any stale queued result so the main process always gets the freshest
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Exception:
                pass
        result_queue.put({'hands': hands})

    detector.close()


# ---------------------------------------------------------------------------
# Swipe detection (runs in main process)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mouse control (runs in main process)
# ---------------------------------------------------------------------------

zot_prev_pos = None


def handle_zot_mouse(landmarks):
    """Move the mouse relative to how much the hand has moved since last frame."""
    global zot_prev_pos
    thumb_tip  = landmarks[4]
    middle_tip = landmarks[12]
    ring_tip   = landmarks[16]
    avg_x = (thumb_tip.x + middle_tip.x + ring_tip.x) / 3
    avg_y = (thumb_tip.y + middle_tip.y + ring_tip.y) / 3
    if zot_prev_pos is not None:
        dx = avg_x - zot_prev_pos[0]
        dy = avg_y - zot_prev_pos[1]
        pyautogui.moveRel(int(dx * SCREEN_W * MOUSE_SENSITIVITY),
                          int(dy * SCREEN_H * MOUSE_SENSITIVITY))
    zot_prev_pos = (avg_x, avg_y)


# ---------------------------------------------------------------------------
# Drawing helpers (run in main process)
# ---------------------------------------------------------------------------

def draw_hand(img, landmarks, handedness_name, w, h, gesture, gesture_color):
    x1, y1, x2, y2 = analyzer.calc_bounding_box(landmarks, w, h)
    cv2.rectangle(img, (x1, y1), (x2, y2), gesture_color, 2)
    cv2.putText(img, f"{handedness_name}: {gesture}", (x1, y1 - 10),
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


# ---------------------------------------------------------------------------
# Main process: capture + display + gesture actions
# ---------------------------------------------------------------------------

def main():
    global zot_prev_pos, last_swipe, last_swipe_time

    # Shared memory: RGB frame for inference
    shm_frame     = multiprocessing.Array(ctypes.c_uint8, FRAME_PIXELS)
    # Counter incremented each time a new frame is written; inference process polls this
    frame_counter = multiprocessing.Value(ctypes.c_ulong, 0)
    # Set to True to tell the inference process to exit cleanly
    stop_flag     = multiprocessing.Value(ctypes.c_bool, False)
    # Inference process sends hand results back here
    result_queue  = multiprocessing.Queue(maxsize=2)

    proc = multiprocessing.Process(
        target=inference_process,
        args=(shm_frame, frame_counter, stop_flag, result_queue),
        daemon=True,
    )
    proc.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    latest_result = {'hands': []}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            now = time.time()

            # Downscale to inference resolution and write into shared memory
            small = cv2.resize(frame, (INFER_W, INFER_H))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            with shm_frame.get_lock():
                np.frombuffer(shm_frame.get_obj(), dtype=np.uint8)[:] = small_rgb.flatten()
            with frame_counter.get_lock():
                frame_counter.value += 1

            # Drain the result queue, keeping only the freshest result
            while True:
                try:
                    latest_result = result_queue.get_nowait()
                except Empty:
                    break

            swipe = None

            for hand_data in latest_result['hands']:
                # Reconstruct landmark-like objects from serialized tuples
                landmarks = [
                    SimpleNamespace(x=d[0], y=d[1], z=d[2])
                    for d in hand_data['landmarks_data']
                ]
                handedness_name = hand_data['handedness_name']
                gesture         = hand_data['gesture']
                color           = hand_data['color']

                draw_hand(frame, landmarks, handedness_name, w, h, gesture, color)

                if gesture == "4":
                    swipe = detect_swipe(landmarks, now)
                elif gesture == "Zot":
                    handle_zot_mouse(landmarks)
                else:
                    wrist_history.clear()
                    zot_prev_pos = None

            if swipe:
                handle_swipe_action(swipe)

            if swipe or (last_swipe and now - last_swipe_time < 0.6):
                draw_swipe_indicator(frame, swipe or last_swipe, w, h)

            num = len(latest_result['hands'])
            cv2.putText(frame, f"Hands: {num}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

            for i, line in enumerate([
                "4: index+middle+ring+pinky extended, thumb closed — swipe to scroll",
                "Zot: index+pinky extended, thumb+middle+ring pinched — moves cursor",
                "Swipe: Hold 4 and move hand L/R/U/D quickly",
            ]):
                cv2.putText(frame, line, (10, h - 60 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

            cv2.putText(frame, "Press Q to quit", (w - 170, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
            cv2.imshow("Gesture Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_flag.value = True
        cap.release()
        cv2.destroyAllWindows()
        proc.join(timeout=3)
        if proc.is_alive():
            proc.terminate()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
