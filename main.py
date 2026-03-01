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

pyautogui.FAILSAFE = False
# Disable pyautogui's 0.1 s pause inserted after every action — big win on slow hardware
pyautogui.PAUSE = 0.0

# ---------------------------------------------------------------------------
# Resolution constants
# ---------------------------------------------------------------------------
# Capture at 640×360 — half the pixels of the old 1280×720, still fine for display
CAP_W,   CAP_H   = 640, 360
# MediaPipe inference input — 160×90 is 16× fewer pixels than the old 640×360
# Hand detection works reliably at this resolution for single-hand gestures
INFER_W, INFER_H = 160, 90
FRAME_PIXELS     = INFER_W * INFER_H * 3

SCREEN_W, SCREEN_H = pyautogui.size()
MOUSE_SENSITIVITY   = 1.5
SCROLL_AMOUNT       = 500

SWIPE_HISTORY_SIZE = 15
SWIPE_MIN_DISTANCE = 0.15
SWIPE_TIME_WINDOW  = 0.5
SWIPE_COOLDOWN     = 0.8

# ---------------------------------------------------------------------------
# Pre-compute connection index arrays once at module load
# The old code rebuilt the HAND_CONNECTIONS list inside draw_hand every call
# ---------------------------------------------------------------------------
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]
_CONN_A = np.array([c[0] for c in _CONNECTIONS], dtype=np.int32)
_CONN_B = np.array([c[1] for c in _CONNECTIONS], dtype=np.int32)

analyzer = HandAnalyzer()


# ---------------------------------------------------------------------------
# Inference child process
# ---------------------------------------------------------------------------

def inference_process(shm_frame, frame_counter, stop_flag, result_queue):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from hand_analyzer import HandAnalyzer
    import numpy as np
    import time

    base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        # VIDEO mode uses inter-frame tracking — far faster than IMAGE mode
        # because it only runs full detection when the hand is lost
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        # Lower thresholds reduce rejected-frame retries on a slow device
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    hand_analyzer = HandAnalyzer()

    # Pre-allocated buffer — avoids a fresh np.copy() / np.empty() every frame
    frame_buf = np.empty(FRAME_PIXELS, dtype=np.uint8)
    last_counter = 0

    while not stop_flag.value:
        current = frame_counter.value
        if current == last_counter:
            time.sleep(0.001)
            continue
        last_counter = current

        # Copy shared memory into pre-allocated buffer — no heap allocation
        np.copyto(frame_buf, np.frombuffer(shm_frame.get_obj(), dtype=np.uint8))

        frame_rgb = frame_buf.reshape(INFER_H, INFER_W, 3)
        ts_ms     = int(time.time() * 1000)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        # detect_for_video is the VIDEO-mode call — timestamp enables tracking
        det_result = detector.detect_for_video(mp_image, ts_ms)

        hands = []
        if det_result.hand_landmarks:
            for landmarks, handedness in zip(det_result.hand_landmarks,
                                             det_result.handedness):
                hand_analyzer.update_state(landmarks, INFER_W, INFER_H)
                state = hand_analyzer.get_state()
                hands.append({
                    # Flat tuple is cheaper to pickle than list-of-3-tuples
                    'lm': tuple(v for lm in landmarks
                                for v in (lm.x, lm.y, lm.z)),
                    'handedness_name':    handedness[0].category_name,
                    'gesture':            state.gesture,
                    'color':              state.color,
                    'openness':           state.openness,
                    'distance_from_cam':  state.distance_from_cam,
                    # openness_history dropped — main process never reads it
                })

        # Flush any stale result then push the fresh one
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Exception:
                pass
        result_queue.put(hands)

    detector.close()


# ---------------------------------------------------------------------------
# Swipe detection
# ---------------------------------------------------------------------------

wrist_history   = deque(maxlen=SWIPE_HISTORY_SIZE)
last_swipe      = ""
last_swipe_time = 0.0


def detect_swipe(lm, current_time):
    global last_swipe, last_swipe_time
    wx, wy = lm[0].x, lm[0].y
    wrist_history.append((wx, wy, current_time))
    if len(wrist_history) < 5 or current_time - last_swipe_time < SWIPE_COOLDOWN:
        return None
    oldest = next((pt for pt in wrist_history
                   if current_time - pt[2] <= SWIPE_TIME_WINDOW), None)
    if oldest is None:
        return None
    dx, dy = wx - oldest[0], wy - oldest[1]
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
    if swipe in ("Swipe Up", "Swipe Right"):
        pyautogui.scroll(SCROLL_AMOUNT)
    elif swipe in ("Swipe Down", "Swipe Left"):
        pyautogui.scroll(-SCROLL_AMOUNT)


# ---------------------------------------------------------------------------
# Mouse control
# ---------------------------------------------------------------------------

zot_prev_pos = None


def handle_zot_mouse(lm):
    global zot_prev_pos
    avg_x = (lm[4].x + lm[12].x + lm[16].x) / 3
    avg_y = (lm[4].y + lm[12].y + lm[16].y) / 3
    if zot_prev_pos is not None:
        dx = avg_x - zot_prev_pos[0]
        dy = avg_y - zot_prev_pos[1]
        pyautogui.moveRel(int(dx * SCREEN_W * MOUSE_SENSITIVITY),
                          int(dy * SCREEN_H * MOUSE_SENSITIVITY))
    zot_prev_pos = (avg_x, avg_y)


# ---------------------------------------------------------------------------
# Drawing helpers — numpy-vectorised
# ---------------------------------------------------------------------------

def draw_hand(img, lm, handedness_name, w, h, gesture, gesture_color):
    """Convert all 21 landmarks to pixel coords in one numpy multiply."""
    xy = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.int32)  # (21,2)

    x1 = int(max(0,  xy[:, 0].min() - 20))
    y1 = int(max(0,  xy[:, 1].min() - 20))
    x2 = int(min(w,  xy[:, 0].max() + 20))
    y2 = int(min(h,  xy[:, 1].max() + 20))
    cv2.rectangle(img, (x1, y1), (x2, y2), gesture_color, 2)
    cv2.putText(img, f"{handedness_name}: {gesture}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)

    # Draw all connections using pre-built index arrays — no per-call list creation
    for (ax, ay), (bx, by) in zip(xy[_CONN_A], xy[_CONN_B]):
        cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)), YELLOW, 2)

    for i, (cx, cy) in enumerate(xy):
        if i in FINGERTIPS:
            cv2.circle(img, (int(cx), int(cy)), 8, RED, -1)
            cv2.circle(img, (int(cx), int(cy)), 8, WHITE, 2)
        else:
            cv2.circle(img, (int(cx), int(cy)), 5, BLUE, -1)


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
            break  # only one direction — exit early
    cv2.putText(img, swipe, (cx - 80, cy - L - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, ORANGE, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global zot_prev_pos, last_swipe, last_swipe_time

    shm_frame     = multiprocessing.Array(ctypes.c_uint8, FRAME_PIXELS)
    frame_counter = multiprocessing.Value(ctypes.c_ulong, 0)
    stop_flag     = multiprocessing.Value(ctypes.c_bool, False)
    result_queue  = multiprocessing.Queue(maxsize=2)

    proc = multiprocessing.Process(
        target=inference_process,
        args=(shm_frame, frame_counter, stop_flag, result_queue),
        daemon=True,
    )
    proc.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Pre-allocate the inference-size write buffer and a persistent view of shared mem
    infer_buf = np.empty((INFER_H, INFER_W, 3), dtype=np.uint8)
    shm_view  = np.frombuffer(shm_frame.get_obj(), dtype=np.uint8)

    latest_hands: list = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            now = time.time()

            # Resize directly into pre-allocated buffer — no new array allocated
            cv2.resize(frame, (INFER_W, INFER_H),
                       dst=infer_buf, interpolation=cv2.INTER_NEAREST)
            # cvtColor runs on the tiny 160×90 image, not the 640×360 frame
            cv2.cvtColor(infer_buf, cv2.COLOR_BGR2RGB, dst=infer_buf)
            shm_view[:] = infer_buf.ravel()
            with frame_counter.get_lock():
                frame_counter.value += 1

            # Drain result queue, keep only freshest
            while True:
                try:
                    latest_hands = result_queue.get_nowait()
                except Empty:
                    break

            swipe = None

            for hand_data in latest_hands:
                raw = hand_data['lm']
                lm  = [SimpleNamespace(x=raw[i*3], y=raw[i*3+1], z=raw[i*3+2])
                       for i in range(21)]

                gesture = hand_data['gesture']
                color   = hand_data['color']

                draw_hand(frame, lm, hand_data['handedness_name'], w, h, gesture, color)

                if gesture == "4":
                    swipe = detect_swipe(lm, now)
                elif gesture == "Zot":
                    handle_zot_mouse(lm)
                else:
                    wrist_history.clear()
                    zot_prev_pos = None

            if swipe:
                handle_swipe_action(swipe)

            if swipe or (last_swipe and now - last_swipe_time < 0.6):
                draw_swipe_indicator(frame, swipe or last_swipe, w, h)

            cv2.putText(frame, f"Hands: {len(latest_hands)}", (10, 30),
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
