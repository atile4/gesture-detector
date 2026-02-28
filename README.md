# Gesture Detector

A Python application for real-time webcam capture using OpenCV.

## Requirements

- Python 3.8+
- Webcam access

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

## Permissions (macOS)

On macOS, you may need to grant camera permissions:
1. Go to **System Preferences → Security & Privacy → Camera**
2. Add Terminal (or your terminal app) to the allowed applications
3. Restart the application

## Features

- Real-time webcam feed display
- Cross-platform support (Windows, macOS, Linux)
- Clean exit with 'q' key
