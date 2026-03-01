# Gesture Detector

A real-time hand gesture recognition app that maps gestures to mouse/keyboard actions using your webcam.

## How It Works

Camera frames are captured in the main process and written to a shared memory buffer. A separate inference process reads the buffer, runs MediaPipe hand landmark detection, classifies the gesture, and sends results back via a queue. The main process handles drawing and input events (mouse movement, scroll, zoom, click).

## Gestures & Controls

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
git clone <your-repo-url>
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

## Troubleshooting

**MediaPipe import errors** — MediaPipe only supports Python 3.11 and earlier. Recreate your venv with the right version:
```bash
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Camera permissions (macOS)** — Go to System Settings → Privacy & Security → Camera and grant access to your terminal app.

**Scroll not working (Linux)** — Make sure your user is in the `input` group or run with appropriate permissions for `evdev`/`UInput`.

## Project Structure

```
gesture-detector/
├── main.py            # Main process: capture, display, input events
├── hand_analyzer.py   # HandAnalyzer: gesture classification, landmark geometry
├── colors.py          # BGR color constants
└── hand_landmarker.task  # MediaPipe model (not included in repo)
```
