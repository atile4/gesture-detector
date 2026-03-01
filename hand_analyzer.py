import math
import numpy as np
from collections import namedtuple, deque

from colors import GREEN, ORANGE, YELLOW, WHITE

FINGERTIPS = {4, 8, 12, 16, 20}

HandState = namedtuple("HandState", [
    "gesture", "color", "openness", "distance_from_cam",
])

# ---------------------------------------------------------------------------
# Landmark indices — module constants, never rebuilt
# ---------------------------------------------------------------------------
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_FINGER_TIP   = {"thumb": 4,  "index": 8,  "middle": 12, "ring": 16, "pinky": 20}
_FINGER_PIP   = {"thumb": 2,  "index": 6,  "middle": 10, "ring": 14, "pinky": 18}
_FINGER_MCP   = {"thumb": 1,  "index": 5,  "middle": 9,  "ring": 13, "pinky": 17}
_OPENNESS_TIPS = np.array([4, 8, 12, 16, 20], dtype=np.int32)

# Frames a gesture must hold before being emitted — raise to reduce flicker,
# lower to reduce latency.
GESTURE_HOLD_FRAMES = 3


class HandAnalyzer:
    """Extracts per-hand properties from a MediaPipe NormalizedLandmark list."""

    def __init__(self):
        self._gesture           = ""
        self._color             = WHITE
        self._openness          = 0.0
        self._distance_from_cam = 1.0
        self._openness_history  = deque(maxlen=10)

        self._lm_xy: np.ndarray = None          # (21,2) float32, reused each frame
        self._diff_buf          = np.empty((5, 2), dtype=np.float32)  # openness buffer

        # Temporal smoothing
        self._candidate_gesture = ""
        self._candidate_color   = WHITE
        self._candidate_count   = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(p1, p2) -> float:
        dx, dy = p1.x - p2.x, p1.y - p2.y
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _is_finger_extended(landmarks, finger: str,
                             hand_label: str = "Right") -> bool:
        """
        Reliable extension check.

        Thumb: chirality-aware — tip must be to the correct side of the MCP
               relative to the display centre of the hand.
        Others: tip y < pip y (higher in image = extended).
        """
        tip_i = _FINGER_TIP[finger]
        pip_i = _FINGER_PIP[finger]

        if finger == "thumb":
            tip = landmarks[tip_i]
            mcp = landmarks[_FINGER_MCP["thumb"]]   # landmark 1
            # Right hand (post-flip): extended thumb moves LEFT of MCP
            # Left  hand (post-flip): extended thumb moves RIGHT of MCP
            if hand_label == "Right":
                return tip.x < mcp.x
            else:
                return tip.x > mcp.x

        return landmarks[tip_i].y < landmarks[pip_i].y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_state(self, landmarks, cam_width: int, cam_height: int,
                     hand_label: str = "Right") -> None:
        """Call once per frame with the raw MediaPipe landmark list."""
        # Rebuild (21,2) array in-place — no list comprehension allocation
        if self._lm_xy is None:
            self._lm_xy = np.empty((21, 2), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            self._lm_xy[i, 0] = lm.x
            self._lm_xy[i, 1] = lm.y

        ext = self._calc_extended_fingers_fast(landmarks, hand_label)
        raw_gesture, raw_color = self._calc_gesture_from_ext(landmarks, ext)

        # Temporal smoothing
        if raw_gesture == self._candidate_gesture:
            self._candidate_count += 1
        else:
            self._candidate_gesture = raw_gesture
            self._candidate_color   = raw_color
            self._candidate_count   = 1

        if self._candidate_count >= GESTURE_HOLD_FRAMES:
            self._gesture = self._candidate_gesture
            self._color   = self._candidate_color

        self._openness          = self._calc_fist_openness_np()
        self._distance_from_cam = self._calc_distance_np(cam_width, cam_height)
        self._openness_history.append(self._openness)

    def get_state(self) -> HandState:
        return HandState(
            gesture=self._gesture,
            color=self._color,
            openness=self._openness,
            distance_from_cam=self._distance_from_cam,
        )

    # ------------------------------------------------------------------
    # Extended finger detection
    # ------------------------------------------------------------------

    def _calc_extended_fingers_fast(self, landmarks,
                                     hand_label: str = "Right") -> dict:
        return {f: self._is_finger_extended(landmarks, f, hand_label)
                for f in _FINGER_NAMES}

    def calc_extended_fingers(self, landmarks,
                               hand_label: str = "Right") -> dict:
        return self._calc_extended_fingers_fast(landmarks, hand_label)

    # ------------------------------------------------------------------
    # Gesture classifier
    # ------------------------------------------------------------------

    def _calc_gesture_from_ext(self, landmarks, ext: dict) -> tuple:
        """
        Most-constrained patterns first to avoid transition flicker.

        "4"   → 4 fingers extended, thumb tucked       (checked first —
                 superset of Zot's index+pinky pattern)
        "Zot" → index + pinky only, thumb explicitly tucked
        "Three" → index + middle + ring, pinky closed
        """
        idx  = ext["index"]
        mid  = ext["middle"]
        ring = ext["ring"]
        pink = ext["pinky"]
        thm  = ext["thumb"]

        if not thm and idx and mid and ring and pink:
            return "4", GREEN

        if idx and pink and not mid and not ring and not thm:
            return "Zot", YELLOW

        if idx and mid and ring and not pink:
            return "Three", ORANGE

        return "", WHITE

    def calc_gesture(self, landmarks, hand_label: str = "Right") -> tuple:
        ext = self._calc_extended_fingers_fast(landmarks, hand_label)
        return self._calc_gesture_from_ext(landmarks, ext)

    # ------------------------------------------------------------------
    # Openness — zero allocation in hot path
    # ------------------------------------------------------------------

    def _calc_fist_openness_np(self) -> float:
        xy    = self._lm_xy
        wrist = xy[0]
        d_ref = xy[9] - wrist
        ref_sq = float(d_ref[0] * d_ref[0] + d_ref[1] * d_ref[1])
        if ref_sq < 1e-12:
            return 0.0
        ref = math.sqrt(ref_sq)
        np.subtract(xy[_OPENNESS_TIPS], wrist, out=self._diff_buf)
        avg = float(np.mean(np.hypot(self._diff_buf[:, 0], self._diff_buf[:, 1])))
        return max(0.0, min(1.0, (avg / ref - 0.8) / 0.7))

    def calc_fist_openness(self, landmarks) -> float:
        if self._lm_xy is None:
            self._lm_xy = np.empty((21, 2), dtype=np.float32)
            for i, lm in enumerate(landmarks):
                self._lm_xy[i, 0] = lm.x
                self._lm_xy[i, 1] = lm.y
        return self._calc_fist_openness_np()

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def _calc_distance_np(self, frame_w: int, frame_h: int,
                          reference_size: float = 150.0) -> float:
        d = (self._lm_xy[0] - self._lm_xy[9]) * \
            np.array([frame_w, frame_h], dtype=np.float32)
        size = math.sqrt(float(d[0] * d[0] + d[1] * d[1]))
        return max(0.3, min(2.0, size / reference_size))

    def calc_distance(self, landmarks, frame_w: int, frame_h: int,
                      reference_size: float = 150.0) -> float:
        if self._lm_xy is None:
            self._lm_xy = np.empty((21, 2), dtype=np.float32)
            for i, lm in enumerate(landmarks):
                self._lm_xy[i, 0] = lm.x
                self._lm_xy[i, 1] = lm.y
        return self._calc_distance_np(frame_w, frame_h, reference_size)

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    def calc_bounding_box(self, landmarks, frame_w: int, frame_h: int,
                          padding: int = 20) -> tuple:
        xy   = self._lm_xy
        mins = xy.min(axis=0)
        maxs = xy.max(axis=0)
        return (
            max(0,        int(mins[0] * frame_w) - padding),
            max(0,        int(mins[1] * frame_h) - padding),
            min(frame_w,  int(maxs[0] * frame_w) + padding),
            min(frame_h,  int(maxs[1] * frame_h) + padding),
        )
