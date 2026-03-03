# Gesture Detector

A real-time hand gesture recognition app that uses gestures captured by the webcam to control mouse actions.

Right now, Gesture Detector supports:

| Gesture | Action |
|---------|--------|
| **Zot** (index + pinky extended) | Mouse movement |
| **Pinky** (pinky only extended) | Left click (when transitioning from Zot) |
| **4** (index + middle + ring + pinky, no thumb) | Scroll |
| **Peace** (index + middle extended) | Zoom (Ctrl + scroll) |

## Requirements

- Python **3.8 through 3.11** (MediaPipe does not support Python 3.12+)
- Webcam
- `hand_landmarker.task` model file in the project root

> **Note:** On Linux, `evdev` is used for low-latency kernel-level input. On macOS/Windows, the app falls back to `pyautogui`.

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/atile4/gesture-detector
cd gesture-detector
```

### 2. Create a virtual environment

**macOS/Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the model

Download `hand_landmarker.task` from [MediaPipe's model page](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and place it in the project root.

## Usage

```bash
python main.py
```

Press **Q** to quit.
