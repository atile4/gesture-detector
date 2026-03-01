import math
from collections import namedtuple, deque

FINGERTIPS = {4, 8, 12, 16, 20}
HandState = namedtuple ("HandState", ["gesture", "color", "openness", "distance_from_cam"
                                      , "openness_history"])

class HandAnalyzer:
    """Extracts per-hand properties from a list of MediaPipe NormalizedLandmark."""

    def __init__(self):
        self._gesture = None
        self._color = None
        self._openness = None
        self._distance_from_cam = None

        self._distance_from_cam_history = deque(maxlen=10)
        self._openness_history = deque(maxlen = 10) # previous 5 openness measurements

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # -----------------------------f------------------------------------- #

    @staticmethod
    def _dist(p1, p2) -> float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    @staticmethod
    def _is_finger_extended(landmarks, finger: str) -> bool:
        fingers = {
            "thumb":  [4, 3, 2, 1],
            "index":  [8, 7, 6, 5],
            "middle": [12, 11, 10, 9],
            "ring":   [16, 15, 14, 13],
            "pinky":  [20, 19, 18, 17],
        }
        tip, dip, pip_, mcp = [landmarks[i] for i in fingers[finger]]
        if finger == "thumb":
            wrist = landmarks[0]
            return abs(tip.x - wrist.x) > abs(mcp.x - wrist.x)
        return tip.y < pip_.y

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update_state(self, landmarks, cam_length, cam_width):
        self._gesture, self._color = self.calc_gesture(landmarks)
        self._openness = self.calc_fist_openness(landmarks)
        self._distance_from_cam = self.calc_distance(landmarks, cam_length, cam_width)


        self._openness_history.append(self._openness)
        self._distance_from_cam_history.append(self._distance_from_cam)

    def get_state(self):
        return HandState(gesture=self._gesture, color = self._color,
                         openness = self._openness, distance_from_cam = self._distance_from_cam,
                         openness_history = self._openness_history)

    def calc_extended_fingers(self, landmarks) -> dict[str, bool]:
        """Returns which fingers are extended."""
        names = ["thumb", "index", "middle", "ring", "pinky"]
        return {f: self._is_finger_extended(landmarks, f) for f in names}

    def calc_gesture(self, landmarks) -> tuple[str, tuple]:
        """Detect static gesture. Returns (label, bgr_color)."""
        from colors import GREEN, RED, ORANGE, YELLOW, WHITE, PURPLE

        ext = self.calc_extended_fingers(landmarks)
        count = sum(ext.values())

        
        if ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]:
            return "Peace", PURPLE

        if ext["thumb"] and not any(ext[f] for f in ["index", "middle", "ring", "pinky"]):
            return "Thumbs Up", GREEN

        if not any(ext[f] for f in ["index", "middle", "ring", "pinky"]):
            if landmarks[4].y > landmarks[2].y:
                return "Thumbs Down", RED

        if count == 0:
            return "Fist", RED
        if count == 5:
            return "Open Palm", GREEN
        if ext["index"] and not ext["middle"] and not ext["ring"] and not ext["pinky"]:
            return "Pointing", ORANGE
        if ext["index"] and ext["pinky"] and not ext["ring"]:
            HORIZONTAL_THRESHOLD = 0.08
            TOUCH_THRESHOLD = 0.06
            middle_is_horizontal = abs(landmarks[12].y - landmarks[9].y) < HORIZONTAL_THRESHOLD
            thumb_touches_pip = (
                self._dist(landmarks[4], landmarks[6]) < TOUCH_THRESHOLD or   # index PIP
                self._dist(landmarks[4], landmarks[10]) < TOUCH_THRESHOLD      # middle PIP
            )
            if middle_is_horizontal and thumb_touches_pip:
                return "Zot", YELLOW
            if not ext["middle"]:
                return "Rock On", YELLOW
        if ext["index"] and ext["middle"] and ext["ring"] and not ext["pinky"]:
            return "Three", ORANGE

        if (not ext["thumb"] and ext["index"] and ext["middle"]
                and ext["ring"] and ext["pinky"]):
            return "4", GREEN

        return "", WHITE

    def calc_fist_openness(self, landmarks) -> float:
        """0.0 = fully closed, 1.0 = fully open."""
        wrist = landmarks[0]
        ref = self._dist(wrist, landmarks[9])
        if ref < 1e-6:
            return 0.0

        avg = sum(self._dist(wrist, landmarks[i]) for i in [4, 8, 12, 16, 20]) / 5
        raw = avg / ref

        return max(0.0, min(1.0, (raw - 0.8) / (1.5 - 0.8)))

    def calc_distance(self, landmarks, frame_w: int, frame_h: int,
                      reference_size: float = 150.0) -> float:
        """Scale factor based on apparent hand size. 1.0 = reference distance."""
        wrist = landmarks[0]
        mcp = landmarks[9]
        dx = (wrist.x - mcp.x) * frame_w
        dy = (wrist.y - mcp.y) * frame_h
        size = math.sqrt(dx ** 2 + dy ** 2)
        return max(0.3, min(2.0, size / reference_size))

    def calc_bounding_box(self, landmarks, frame_w: int, frame_h: int,
                          padding: int = 20) -> tuple[int, int, int, int]:
        """Returns (x1, y1, x2, y2) pixel bounding box."""
        xs = [lm.x * frame_w for lm in landmarks]
        ys = [lm.y * frame_h for lm in landmarks]
        return (
            max(0, int(min(xs)) - padding),
            max(0, int(min(ys)) - padding),
            min(frame_w, int(max(xs)) + padding),
            min(frame_h, int(max(ys)) + padding),
        )
