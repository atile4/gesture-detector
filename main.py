import multiprocessing
import ctypes
import numpy as np
import cv2
import time
import pyautogui
from queue import Empty

from hand_analyzer import HandAnalyzer, FINGERTIPS
from colors import *

# ---------------------------------------------------------------------------
# evdev virtual device setup
# Declared at module level so the UInput device persists for the process
# lifetime and is shared by scroll + mouse functions.
# ---------------------------------------------------------------------------
try:
    from evdev import UInput, ecodes as EC
    _UINPUT_CAP = {
        EC.EV_REL: [
            EC.REL_X,
            EC.REL_Y,
            EC.REL_WHEEL,
            EC.REL_HWHEEL,
        ],
        EC.EV_KEY: [
            EC.BTN_LEFT,
            EC.BTN_RIGHT,
            EC.BTN_MIDDLE,
        ],
    }
    _ui = UInput(_UINPUT_CAP, name="gesture-control", version=0x1)
    _EVDEV_OK = True
except Exception as _evdev_err:
    print(f"[WARN] evdev UInput unavailable ({_evdev_err}), falling back to pyautogui")
    _ui = None
    _EVDEV_OK = False

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.0

# ---------------------------------------------------------------------------
# Resolution constants
# ---------------------------------------------------------------------------
CAP_W,   CAP_H   = 640, 360
INFER_W, INFER_H = 160, 90
FRAME_PIXELS     = INFER_W * INFER_H * 3

SCREEN_W, SCREEN_H = pyautogui.size()

# ---------------------------------------------------------------------------
# Control settings
# ---------------------------------------------------------------------------
MOUSE_SENSITIVITY  = 1.5
DEAD_ZONE          = 0.005

# Scroll: hand moves ~0.01–0.05 normalised units per frame at a natural speed.
# SCROLL_SENSITIVITY scales that into REL_WHEEL clicks (integers).
# REL_WHEEL=1 is one detent on a standard scroll wheel.
# A value of 30 means a full hand sweep (~0.3 units) produces 9 wheel clicks.
SCROLL_SENSITIVITY = 80
SCROLL_DEAD_ZONE   = 0.003

# ---------------------------------------------------------------------------
# Pre-computed connection index arrays
# ---------------------------------------------------------------------------
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]
_CONN_A = np.array([c[0] for c in _CONNECTIONS], dtype=np.int32)
_CONN_B = np.array([c[1] for c in _CONNECTIONS], dtype=np.int32)


# ---------------------------------------------------------------------------
# Input helpers — evdev fast path with pyautogui fallback
# ---------------------------------------------------------------------------

def _scroll(clicks: int):
    """
    Inject a scroll-wheel event.
    clicks > 0 = scroll up, clicks < 0 = scroll down.
    Uses REL_WHEEL which is a standard integer detent count.
    The write() call is non-blocking — returns in microseconds.
    """
    if clicks == 0:
        return
    if _EVDEV_OK:
        _ui.write(EC.EV_REL, EC.REL_WHEEL, clicks)
        _ui.syn()
    else:
        # pyautogui fallback — may be slow on some systems
        pyautogui.scroll(clicks * 3)


def _move_rel(dx: int, dy: int):
    """
    Inject a relative mouse-movement event.
    Both writes are batched into a single syn() call so they appear atomic.
    """
    if dx == 0 and dy == 0:
        return
    if _EVDEV_OK:
        _ui.write(EC.EV_REL, EC.REL_X, dx)
        _ui.write(EC.EV_REL, EC.REL_Y, dy)
        _ui.syn()
    else:
        pyautogui.moveRel(dx, dy)


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
    detector      = vision.HandLandmarker.create_from_options(options)
    hand_analyzer = HandAnalyzer()

    shm_view  = np.frombuffer(shm_frame.get_obj(), dtype=np.uint8)
    frame_buf = np.empty(FRAME_PIXELS, dtype=np.uint8)
    last_counter = 0

    while not stop_flag.value:
        current = frame_counter.value
        if current == last_counter:
            time.sleep(0.001)
            continue
        last_counter = current

        with shm_frame.get_lock():
            np.copyto(frame_buf, shm_view)

        frame_rgb  = frame_buf.reshape(INFER_H, INFER_W, 3)
        ts_ms      = int(time.time() * 1000)
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        det_result = detector.detect_for_video(mp_image, ts_ms)

        hands = []
        if det_result.hand_landmarks:
            for landmarks, handedness in zip(det_result.hand_landmarks,
                                             det_result.handedness):
                hand_label = handedness[0].category_name
                hand_analyzer.update_state(landmarks, INFER_W, INFER_H, hand_label)
                state = hand_analyzer.get_state()
                hands.append({
                    'lm':              tuple(v for lm in landmarks
                                            for v in (lm.x, lm.y, lm.z)),
                    'handedness_name': hand_label,
                    'gesture':         state.gesture,
                    'color':           state.color,
                    'openness':        state.openness,
                    'distance':        state.distance_from_cam,
                })

        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Exception:
                pass
        result_queue.put(hands)

    detector.close()


# ---------------------------------------------------------------------------
# Scroll handler  (called directly from main loop — no thread needed)
# ---------------------------------------------------------------------------

scroll_prev_y = None


def handle_4_scroll(lm_flat):
    """
    Maps vertical wrist movement to integer REL_WHEEL detents.
    dy is normalised (0–1 range), so multiply by SCROLL_SENSITIVITY
    and round to the nearest integer wheel click.
    """
    global scroll_prev_y
    wy = lm_flat[1]   # wrist y: landmark 0, y component = index 1
    if scroll_prev_y is not None:
        dy = wy - scroll_prev_y
        if abs(dy) > SCROLL_DEAD_ZONE:
            # dy > 0 = hand moved down → scroll down = negative wheel
            clicks = -round(dy * SCROLL_SENSITIVITY)
            _scroll(clicks)
    scroll_prev_y = wy


# ---------------------------------------------------------------------------
# Mouse handler  (called directly from main loop — no thread needed)
# ---------------------------------------------------------------------------

zot_prev_pos = None


def handle_zot_mouse(lm_flat):
    global zot_prev_pos
    avg_x = (lm_flat[4*3] + lm_flat[12*3] + lm_flat[16*3]) / 3
    avg_y = (lm_flat[4*3+1] + lm_flat[12*3+1] + lm_flat[16*3+1]) / 3
    if zot_prev_pos is not None:
        dx = avg_x - zot_prev_pos[0]
        dy = avg_y - zot_prev_pos[1]
        if abs(dx) > DEAD_ZONE or abs(dy) > DEAD_ZONE:
            _move_rel(int(dx * SCREEN_W * MOUSE_SENSITIVITY),
                      int(dy * SCREEN_H * MOUSE_SENSITIVITY))
    zot_prev_pos = (avg_x, avg_y)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_hand(img, lm_flat, handedness_name, w, h, gesture, gesture_color):
    lm_arr = np.array(lm_flat, dtype=np.float32).reshape(21, 3)
    xy = (lm_arr[:, :2] * np.array([w, h], dtype=np.float32)).astype(np.int32)

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
    cv2.putText(img, "SCROLLING", (w // 2 - 60, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global zot_prev_pos, scroll_prev_y

    if _EVDEV_OK:
        print("[INPUT] evdev UInput device ready — kernel-level input, no X latency")
    else:
        print("[INPUT] WARNING: using pyautogui fallback — scroll may be slow")

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

            cv2.resize(frame, (INFER_W, INFER_H),
                       dst=infer_buf, interpolation=cv2.INTER_NEAREST)
            cv2.cvtColor(infer_buf, cv2.COLOR_BGR2RGB, dst=infer_buf)
            with shm_frame.get_lock():
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
                lm_flat        = hand_data['lm']
                gesture        = hand_data['gesture']
                color          = hand_data['color']
                active_gesture = gesture

                draw_hand(frame, lm_flat, hand_data['handedness_name'],
                          w, h, gesture, color)

                if gesture == "4":
                    handle_4_scroll(lm_flat)
                    zot_prev_pos = None
                elif gesture == "Zot":
                    handle_zot_mouse(lm_flat)
                    scroll_prev_y = None
                else:
                    zot_prev_pos  = None
                    scroll_prev_y = None

            if active_gesture == "4":
                draw_scroll_indicator(frame, w, h)

            cv2.putText(frame, f"Hands: {len(latest_hands)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            for i, line in enumerate([
                "4: index+middle+ring+pinky extended — drag up/down to scroll",
                "Zot: index+pinky, others closed — moves cursor",
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
        if _EVDEV_OK:
            _ui.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()

