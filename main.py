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
pyautogui.PAUSE = 0.0

# ---------------------------------------------------------------------------
# Resolution constants
# ---------------------------------------------------------------------------
CAP_W,   CAP_H   = 640, 360
INFER_W, INFER_H = 160, 90
FRAME_PIXELS     = INFER_W * INFER_H * 3

SCREEN_W, SCREEN_H = pyautogui.size()

# ---------------------------------------------------------------------------
# Mouse (Zot) settings
# ---------------------------------------------------------------------------
MOUSE_SENSITIVITY = 1.5
DEAD_ZONE         = 0.005

# ---------------------------------------------------------------------------
# Continuous scroll (4 gesture) settings
# ---------------------------------------------------------------------------
SCROLL_SENSITIVITY = 8000  # higher = faster scroll per unit of hand movement
SCROLL_DEAD_ZONE   = 0.005  # ignore tiny movements to prevent jitter

# ---------------------------------------------------------------------------
# Pre-compute connection index arrays once at module load
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
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    hand_analyzer = HandAnalyzer()

    frame_buf = np.empty(FRAME_PIXELS, dtype=np.uint8)
    last_counter = 0

    while not stop_flag.value:
        current = frame_counter.value
        if current == last_counter:
            time.sleep(0.001)
            continue
        last_counter = current

        np.copyto(frame_buf, np.frombuffer(shm_frame.get_obj(), dtype=np.uint8))

        frame_rgb = frame_buf.reshape(INFER_H, INFER_W, 3)
        ts_ms     = int(time.time() * 1000)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        det_result = detector.detect_for_video(mp_image, ts_ms)

        hands = []
        if det_result.hand_landmarks:
            for landmarks, handedness in zip(det_result.hand_landmarks,
                                             det_result.handedness):
                hand_analyzer.update_state(landmarks, INFER_W, INFER_H)
                state = hand_analyzer.get_state()
                hands.append({
                    'lm': tuple(v for lm in landmarks
                                for v in (lm.x, lm.y, lm.z)),
                    'handedness_name':   handedness[0].category_name,
                    'gesture':           state.gesture,
                    'color':             state.color,
                    'openness':          state.openness,
                    'distance_from_cam': state.distance_from_cam,
                })

        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Exception:
                pass
        result_queue.put(hands)

    detector.close()


# ---------------------------------------------------------------------------
# Continuous scroll — trackpad style
# ---------------------------------------------------------------------------

scroll_prev_pos = None  # now stores (x, y) tuple


def handle_scroll(lm):
    """Scroll proportionally to hand movement while 4 is held.
    Vertical movement scrolls up/down, horizontal scrolls left/right."""
    global scroll_prev_pos

    wx = lm[0].x  # wrist x position
    wy = lm[0].y  # wrist y position

    if scroll_prev_pos is not None:
        dx = wx - scroll_prev_pos[0]
        dy = wy - scroll_prev_pos[1]

        if abs(dy) > SCROLL_DEAD_ZONE:
            # dy > 0 = hand moved down → scroll down (negative)
            pyautogui.scroll(int(-dy * SCROLL_SENSITIVITY))

        if abs(dx) > SCROLL_DEAD_ZONE:
            # dx > 0 = hand moved right → scroll right (positive)
            pyautogui.hscroll(int(dx * SCROLL_SENSITIVITY))

    scroll_prev_pos = (wx, wy)


# ---------------------------------------------------------------------------
# Mouse control (Zot)
# ---------------------------------------------------------------------------

zot_prev_pos = None


def handle_zot_mouse(lm):
    global zot_prev_pos
    avg_x = (lm[4].x + lm[12].x + lm[16].x) / 3
    avg_y = (lm[4].y + lm[12].y + lm[16].y) / 3
    if zot_prev_pos is not None:
        dx = avg_x - zot_prev_pos[0]
        dy = avg_y - zot_prev_pos[1]
        if abs(dx) > DEAD_ZONE or abs(dy) > DEAD_ZONE:
            pyautogui.moveRel(int(dx * SCREEN_W * MOUSE_SENSITIVITY),
                              int(dy * SCREEN_H * MOUSE_SENSITIVITY))
    zot_prev_pos = (avg_x, avg_y)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_hand(img, lm, handedness_name, w, h, gesture, gesture_color):
    xy = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.int32)

    x1 = int(max(0, xy[:, 0].min() - 20))
    y1 = int(max(0, xy[:, 1].min() - 20))
    x2 = int(min(w, xy[:, 0].max() + 20))
    y2 = int(min(h, xy[:, 1].max() + 20))
    cv2.rectangle(img, (x1, y1), (x2, y2), gesture_color, 2)
    cv2.putText(img, f"{handedness_name}: {gesture}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)

    for (ax, ay), (bx, by) in zip(xy[_CONN_A], xy[_CONN_B]):
        cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)), YELLOW, 2)

    for i, (cx, cy) in enumerate(xy):
        if i in FINGERTIPS:
            cv2.circle(img, (int(cx), int(cy)), 8, RED, -1)
            cv2.circle(img, (int(cx), int(cy)), 8, WHITE, 2)
        else:
            cv2.circle(img, (int(cx), int(cy)), 5, BLUE, -1)


def draw_scroll_indicator(img, w, h):
    """Show a small indicator when 4 scroll is active."""
    cv2.putText(img, "SCROLLING", (w // 2 - 60, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global zot_prev_pos, scroll_prev_pos

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

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

            cv2.resize(frame, (INFER_W, INFER_H),
                       dst=infer_buf, interpolation=cv2.INTER_NEAREST)
            cv2.cvtColor(infer_buf, cv2.COLOR_BGR2RGB, dst=infer_buf)
            shm_view[:] = infer_buf.ravel()
            with frame_counter.get_lock():
                frame_counter.value += 1

            while True:
                try:
                    latest_hands = result_queue.get_nowait()
                except Empty:
                    break

            active_gesture = None

            for hand_data in latest_hands:
                raw = hand_data['lm']
                lm  = [SimpleNamespace(x=raw[i*3], y=raw[i*3+1], z=raw[i*3+2])
                       for i in range(21)]

                gesture = hand_data['gesture']
                color   = hand_data['color']
                active_gesture = gesture

                draw_hand(frame, lm, hand_data['handedness_name'], w, h, gesture, color)

                if gesture == "4":
                    handle_scroll(lm)
                    zot_prev_pos = None
                elif gesture == "Zot":
                    handle_zot_mouse(lm)
                    scroll_prev_pos = None
                else:
                    zot_prev_pos    = None
                    scroll_prev_pos = None

            # Show scrolling indicator when 4 is active
            if active_gesture == "4":
                draw_scroll_indicator(frame, w, h)

            cv2.putText(frame, f"Hands: {len(latest_hands)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            for i, line in enumerate([
                "4: index+middle+ring+pinky extended — drag up/down to scroll",
                "Zot: index+pinky extended, thumb+middle+ring pinched — moves cursor",
            ]):
                cv2.putText(frame, line, (10, h - 40 + i * 20),
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
