# Gesture Detector

A Python application for real-time webcam capture using OpenCV.

## Requirements

- Python **3.8 through 3.11** (MediaPipe and other packages may not yet support Python 3.12+ or 3.14)
- Webcam access

> ⚠️ **Important:** Users running Python 3.12 or 3.14 have reported import errors with MediaPipe (`module 'mediapipe' has no attribute 'solutions'`). If you encounter such issues, downgrade your interpreter to 3.11 or earlier.

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd gesture-detector
```

### 2. Create a virtual environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

**On macOS/Linux:**
```bash
./venv/bin/python main.py
```

**On Windows:**
```bash
venv\Scripts\python main.py
```

Press **'q'** to quit the application.

### If you hit import errors

MediaPipe's `solutions` API and other binaries are built for specific Python versions. On macOS, Windows and Linux, the pip wheel currently supports up through Python 3.11. Running the script with Python 3.12+ (including your system's 3.14) can lead to `ImportError`.

To fix it:

```bash
# remove existing environment
rm -rf venv
# install with specific python binary
python3.11 -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Permissions (macOS)

On macOS, you may need to grant camera permissions:
1. Go to **System Preferences → Security & Privacy → Camera**
2. Add Terminal (or your terminal app) to the allowed applications
3. Restart the application

## Features

- Real-time webcam feed display
- Cross-platform support (Windows, macOS, Linux)
- Clean exit with 'q' key
