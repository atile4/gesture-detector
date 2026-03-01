import math
import numpy as np
from collections import namedtuple, deque

FINGERTIPS = {4, 8, 12, 16, 20}
HandState = namedtuple("HandState", [
    "gesture", "color", "openness", "distance_from_cam", "openness_history", "gesture_history"
])

# ---------------------------------------------------------------------------
# Pre-built finger index arrays — constructed once at import, never rebuilt
# ---------------------------------------------------------------------------
# tip, pip indices for each finger (used by _is_finger_extended)
_FINGER_TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
_FINGER_PIP = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# Fingertip + wrist landmark indices for openness calc — vectorised in numpy
_OPENNESS_TIPS = np.array([4, 8, 12, 16, 20], dtype=np.int32)


class HandAnalyzer:
    """Extracts per-hand properties from a list of MediaPipe NormalizedLandmark."""

    def __init__(self):
        self._gesture         = ""
        self._color           = None
        self._openness        = 0.0
        self._distance_from_cam = 1.0
        self._openness_history = deque(maxlen=10)
        self._gesture_history = deque(maxlen=10)

        # Cache the last landmark array so helpers don't recompute it
        self._lm_xy: np.ndarray = None   # shape (21, 2), float32

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(p1, p2) -> float:
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return math.sqrt(dx * dx + dy * dy)   # faster than **2 on CPython

    @staticmethod
    def _is_finger_extended(landmarks, finger: str) -> bool:
        # Uses pre-built index constants — no dict construction per call
        tip_i = _FINGER_TIP[finger]
        pip_i = _FINGER_PIP[finger]
        if finger == "thumb":
            wrist = landmarks[0]
            tip   = landmarks[tip_i]
            mcp   = landmarks[1]   # thumb MCP
            return abs(tip.x - wrist.x) > abs(mcp.x - wrist.x)
        return landmarks[tip_i].y < landmarks[pip_i].y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_state(self, landmarks, cam_width, cam_height):
        """Call once per frame with the raw MediaPipe landmark list."""
        # Build numpy array once; reuse across all calculations this frame
        self._lm_xy = np.array([[lm.x, lm.y] for lm in landmarks],
                               dtype=np.float32)

        # Compute extended fingers once and share with gesture detection
        ext = self._calc_extended_fingers_fast(landmarks)

        self._gesture, self._color = self._calc_gesture_from_ext(landmarks, ext)
        self._openness             = self._calc_fist_openness_np()
        self._distance_from_cam    = self._calc_distance_np(cam_width, cam_height)
        self._openness_history.append(self._openness)
        self._gesture_history.append(self._gesture)

    def get_state(self) -> HandState:
        # Pass the deque directly — avoids list() copy every frame
        return HandState(
            gesture=self._gesture,
            color=self._color,
            openness=self._openness,
            distance_from_cam=self._distance_from_cam,
            openness_history=self._openness_history,
            gesture_history=self._gesture_history
        )

    # ------------------------------------------------------------------
    # Extended finger detection
    # ------------------------------------------------------------------

    def _calc_extended_fingers_fast(self, landmarks) -> dict:
        """Compute extension for all 5 fingers in one pass."""
        return {f: self._is_finger_extended(landmarks, f) for f in _FINGER_NAMES}

    def calc_extended_fingers(self, landmarks) -> dict:
        """Public alias — kept for API compatibility."""
        return self._calc_extended_fingers_fast(landmarks)

    # ------------------------------------------------------------------
    # Gesture
    # ------------------------------------------------------------------

    def _calc_gesture_from_ext(self, landmarks, ext: dict) -> tuple:
        """Classify gesture given pre-computed ext dict. No repeated finger checks."""
        from colors import GREEN, RED, ORANGE, YELLOW, WHITE, PURPLE

        idx  = ext["index"]
        mid  = ext["middle"]
        ring = ext["ring"]
        pink = ext["pinky"]
        thm  = ext["thumb"]
       
        if idx and pink and not ring and not mid:
           return "Zot", YELLOW
           
        if idx and mid and ring and not pink:
            return "Three", ORANGE

        if not thm and idx and mid and ring and pink:
            return "4", GREEN

        elif pink and not idx and not mid and not ring:
            return "Pinky", RED

        return "", WHITE

    def calc_gesture(self, landmarks) -> tuple:
        """Public alias — recomputes ext internally (API compatibility)."""
        ext = self._calc_extended_fingers_fast(landmarks)
        return self._calc_gesture_from_ext(landmarks, ext)

    # ------------------------------------------------------------------
    # Openness — vectorised numpy, no Python loop
    # ------------------------------------------------------------------

    def _calc_fist_openness_np(self) -> float:
        """Uses self._lm_xy (set in update_state). No _dist calls."""
        xy    = self._lm_xy
        wrist = xy[0]                              # (2,)
        ref   = float(np.hypot(*(xy[9] - wrist)))  # wrist→middle MCP
        if ref < 1e-6:
            return 0.0
        # Distance from wrist to each fingertip in one vectorised op
        diffs = xy[_OPENNESS_TIPS] - wrist         # (5, 2)
        avg   = float(np.mean(np.hypot(diffs[:, 0], diffs[:, 1])))
        raw   = avg / ref
        return max(0.0, min(1.0, (raw - 0.8) / (1.5 - 0.8)))

    def calc_fist_openness(self, landmarks) -> float:
        """Public alias — rebuilds array if called standalone."""
        if self._lm_xy is None or len(self._lm_xy) != len(landmarks):
            self._lm_xy = np.array([[lm.x, lm.y] for lm in landmarks],
                                   dtype=np.float32)
        return self._calc_fist_openness_np()

    # ------------------------------------------------------------------
    # Distance — reuses self._lm_xy
    # ------------------------------------------------------------------

    def _calc_distance_np(self, frame_w: int, frame_h: int,
                          reference_size: float = 150.0) -> float:
        xy = self._lm_xy
        d  = (xy[0] - xy[9]) * np.array([frame_w, frame_h], dtype=np.float32)
        size = float(np.hypot(d[0], d[1]))
        return max(0.3, min(2.0, size / reference_size))

    def calc_distance(self, landmarks, frame_w: int, frame_h: int,
                      reference_size: float = 150.0) -> float:
        """Public alias."""
        if self._lm_xy is None:
            self._lm_xy = np.array([[lm.x, lm.y] for lm in landmarks],
                                   dtype=np.float32)
        return self._calc_distance_np(frame_w, frame_h, reference_size)

    # ------------------------------------------------------------------
    # Bounding box — vectorised
    # ------------------------------------------------------------------

    def calc_bounding_box(self, landmarks, frame_w: int, frame_h: int,
                          padding: int = 20) -> tuple:
        """Returns (x1, y1, x2, y2) pixel bounding box."""
        if self._lm_xy is not None and len(self._lm_xy) == len(landmarks):
            xy = self._lm_xy
        else:
            xy = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
        mins = xy.min(axis=0)
        maxs = xy.max(axis=0)
        x1 = max(0,        int(mins[0] * frame_w) - padding)
        y1 = max(0,        int(mins[1] * frame_h) - padding)
        x2 = min(frame_w,  int(maxs[0] * frame_w) + padding)
        y2 = min(frame_h,  int(maxs[1] * frame_h) + padding)
        return x1, y1, x2, y2
