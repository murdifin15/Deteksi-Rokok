# Real-Time Cigarette Detection System (YOLO11)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-based real-time object detection and tracking system designed to identify cigarettes and smoking activities from live camera feeds (webcam/CCTV).

---

## Overview

This project provides an automated computer vision solution for monitoring smoke-free areas such as educational institutions, healthcare facilities, fuel stations, hazardous industrial environments, and public buildings. 

By integrating **Ultralytics YOLO11** with **ByteTrack**, the system achieves low-latency inference and stable multi-frame object tracking suitable for deployment on standard workstation CPUs and edge devices.

---

## Features

- **Synchronous Real-Time Detection**: Frame reading and model inference are executed synchronously to ensure bounding box alignment without display lag.
- **Object Tracking (ByteTrack)**: Integrated tracking algorithm to maintain object continuity and reduce detection flickering across consecutive frames.
- **Multi-Camera Support**: Automatic detection and fallback between internal laptop cameras and external USB webcams.
- **Dynamic Sensitivity Adjustment**: Real-time confidence threshold tuning using keyboard inputs during runtime.
- **Snapshot Capture**: On-demand image capture saved with timestamped filenames for evidence logging.
- **One-Click Launcher**: Batch script (`run.bat`) for quick startup on Windows environments.

---

## Tech Stack

- **Programming Language**: Python 3.10+
- **Object Detection Model**: Ultralytics YOLO11
- **Tracking Algorithm**: ByteTrack
- **Computer Vision**: OpenCV
- **Inference Runtime**: PyTorch, NumPy

---

## Directory Structure

```text
Deteksi-Rokok/
├── best model.pt          # Pre-trained YOLO11 model weights (~5.4 MB)
├── captures/              # Directory for saved detection snapshots
│   └── .gitkeep
├── detect_webcam.py       # Main real-time inference and tracking script
├── requirements.txt       # Python package dependencies
├── run.bat                # Interactive Windows launcher
├── .gitignore             # Git ignore patterns
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/murdifin15/Deteksi-Rokok.git
cd Deteksi-Rokok
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Option A: Windows Launcher
Run the batch file directly or via terminal:
```cmd
run.bat
```

### Option B: Command Line Interface (CLI)

- **Run with external webcam (source 1):**
  ```bash
  python detect_webcam.py --source 1
  ```

- **Run with default/internal camera (source 0):**
  ```bash
  python detect_webcam.py --source 0
  ```

- **Run with custom confidence threshold and image size:**
  ```bash
  python detect_webcam.py --source 1 --conf 0.35 --imgsz 320
  ```

### CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | str | `best model.pt` | Path to the trained YOLO model file |
| `--source` | str | `1` | Video source index or file path |
| `--conf` | float | `0.25` | Confidence threshold for object detection |
| `--iou` | float | `0.45` | IoU threshold for Non-Maximum Suppression |
| `--imgsz` | int | `320` | Input image resolution for inference |
| `--label` | str | `Rokok` | Display label for detected class |
| `--test` | flag | `False` | Run self-test mode with dummy image |

---

## Keyboard Controls

| Key | Action |
| :---: | :--- |
| `Q` / `ESC` | Exit application |
| `S` | Save current frame to `captures/` folder |
| `+` / `=` | Increase confidence threshold (+0.05) |
| `-` / `_` | Decrease confidence threshold (-0.05) |

---

## License

This project is licensed under the [MIT License](LICENSE).
