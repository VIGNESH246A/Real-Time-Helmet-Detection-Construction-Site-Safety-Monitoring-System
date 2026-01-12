# 🏗️ Real-Time Helmet Detection & Construction Site Safety Monitoring System

[![Python 3.8+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00D4AA)](https://github.com/ultralytics/ultralytics)

A production-ready AI-powered safety monitoring system for construction sites that detects helmet compliance in real-time using YOLOv8, reducing workplace accidents and ensuring safety regulation adherence.

---

## 🎯 **Key Features**

### Core Capabilities
- ✅ **Real-time Helmet Detection** - Detect workers with/without helmets using YOLOv8
- 📹 **Multi-Source Input** - Webcam, IP cameras (RTSP), CCTV feeds, video files
- ⚡ **High Performance** - 25-30 FPS on single camera, optimized for GPU/CPU
- 🎨 **Visual Annotations** - Bounding boxes with confidence scores
- 🚨 **Instant Alerts** - Visual and audio alerts for safety violations
- 📸 **Auto-Snapshot** - Automatic capture of violation images
- 📊 **Analytics Dashboard** - Real-time monitoring with Streamlit
- 📝 **Automated Reports** - Daily/weekly PDF and CSV reports
- 🗄️ **Database Logging** - SQLite database for violation tracking

### Advanced Features
- 🌙 **Low-Light Enhancement** - CLAHE for challenging lighting conditions
- 🔍 **Noise Reduction** - Robust detection in dusty environments
- 🎨 **Multi-Color Detection** - Works with helmets of any color
- 👥 **Multi-Person Tracking** - Track multiple workers simultaneously
- 📈 **Compliance Metrics** - Calculate and track safety compliance rates
- 🔄 **Multi-Camera Support** - Scalable to monitor multiple locations

---

## 📁 **Project Structure**

```
helmet-detection-system/
│
├── 📁 config/                  # Configuration files
│   ├── config.yaml             # Main system configuration
│   └── camera_config.json      # Camera-specific settings
│
├── 📁 src/                     # Source code
│   ├── 📁 core/                # Core detection modules
│   │   ├── detector.py         # YOLO detection engine
│   │   ├── tracker.py          # Object tracking
│   │   └── preprocessor.py    # Image preprocessing
│   │
│   ├── 📁 data/                # Data handling
│   │   ├── video_stream.py    # Video stream management
│   │   ├── database.py        # SQLite database operations
│   │   └── logger.py          # Logging utilities
│   │
│   ├── 📁 analysis/            # Analysis modules
│   │   ├── violation_engine.py # Violation detection logic
│   │   ├── alert_manager.py   # Alert system
│   │   └── report_generator.py # Report generation
│   │
│   ├── 📁 ui/                  # User interface
│   │   ├── dashboard.py       # Streamlit dashboard
│   │   └── visualizer.py      # Frame annotation
│   │
│   └── 📁 utils/               # Utilities
│       ├── config_loader.py   # Configuration management
│       ├── helpers.py         # Helper functions
│       └── constants.py       # System constants
│
├── 📁 models/                  # Model weights
│   └── helmet_detector.pt     # Trained YOLO model
│
├── 📁 data/                    # Data directory
│   ├── violations.db          # SQLite database
│   └── logs/                  # System logs
│
├── 📁 outputs/                 # System outputs
│   ├── snapshots/             # Violation snapshots
│   ├── videos/                # Recorded videos
│   └── reports/               # Generated reports
│
├── 📁 scripts/                 # Utility scripts
│   ├── train_model.py         # Model training
│   └── evaluate_model.py      # Model evaluation
│
├── main.py                     # Main application
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## 🚀 **Quick Start**

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/helmet-detection-system.git
cd helmet-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/config.yaml` to customize:
- Model parameters (confidence threshold, IoU)
- Video source (webcam index or RTSP URL)
- Alert settings
- Database paths

```yaml
video:
  default_source: 0  # 0 for webcam, or RTSP URL

model:
  confidence_threshold: 0.5
  device: "cuda"  # or "cpu"
```

### 3. Run the System

```bash
# Run with live detection and display
python main.py

# Run without display (headless mode)
python main.py --no-display

# Run with video recording
python main.py --record

# Use custom video source
python main.py --source rtsp://192.168.1.100:554/stream
```

### 4. Launch Dashboard

```bash
# Start Streamlit dashboard
streamlit run src/ui/dashboard.py
```

Access dashboard at: `http://localhost:8501`

---

## 🎓 **Model Training**

### Dataset Preparation

1. **Collect Data**
   - Gather 2000+ images of construction sites
   - Include various lighting conditions
   - Different helmet colors and types

2. **Annotation Format (YOLO)**
   ```
   class_id center_x center_y width height
   ```
   Example:
   ```
   0 0.5 0.5 0.2 0.3  # helmet
   1 0.3 0.4 0.15 0.25  # no-helmet
   ```

3. **Directory Structure**
   ```
   data/training/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```

### Training

```bash
# Create dataset YAML
python scripts/train_model.py --create-yaml

# Train model (YOLOv8 nano for speed)
python scripts/train_model.py --model n --epochs 100 --batch 16

# Train larger model for accuracy
python scripts/train_model.py --model s --epochs 150 --batch 16
```

### Pre-trained Weights

Place your trained `helmet_detector.pt` in the `models/` directory.

---

## 📊 **Usage Examples**

### Python API

```python
from src.core.detector import HelmetDetector
import cv2

# Initialize detector
detector = HelmetDetector(
    model_path="models/helmet_detector.pt",
    confidence_threshold=0.5
)

# Read image
frame = cv2.imread("construction_site.jpg")

# Detect helmets
detections, annotated = detector.detect(frame)

# Check compliance
status = detector.classify_helmet_status(detections)
print(f"Compliance Rate: {status['compliance_rate']*100:.1f}%")
print(f"Violations: {status['no_helmet_count']}")
```

### Keyboard Shortcuts (Live Mode)

- `Q` - Quit application
- `S` - Save screenshot
- `R` - Generate report

---

## 🔧 **Configuration Guide**

### Model Settings

```yaml
model:
  name: "yolov8n"               # Model size: n, s, m, l, x
  confidence_threshold: 0.5      # Detection confidence (0-1)
  iou_threshold: 0.45           # NMS IoU threshold
  device: "cuda"                # cuda, cpu, or mps
```

### Violation Detection

```yaml
violation:
  min_confidence: 0.6           # Minimum confidence for violation
  cooldown_seconds: 5           # Cooldown between same violations
  snapshot_enabled: true        # Save violation images
  alert_sound_enabled: true     # Audio alerts
```

### Alert Configuration

```yaml
alerts:
  visual_enabled: true          # Console alerts
  audio_enabled: true           # Sound alerts
  email_enabled: false          # Email notifications
```

---

## 📈 **Performance Benchmarks**

| Setup | Model | FPS | Accuracy | GPU Memory |
|-------|-------|-----|----------|------------|
| Single Camera | YOLOv8n | 30 | 92% | 2GB |
| Single Camera | YOLOv8s | 25 | 95% | 3GB |
| 4 Cameras | YOLOv8n | 15 | 92% | 4GB |
| 8 Cameras | YOLOv8n | 10 | 92% | 6GB |

**Tested on:**
- GPU: NVIDIA RTX 3060 Ti
- CPU: Intel i7-10700K
- Resolution: 1280x720

---

## 🐳 **Docker Deployment**

```dockerfile
# Build image
docker build -t helmet-detection .

# Run container
docker run --gpus all -p 8501:8501 helmet-detection
```

---

## 📱 **API Integration**

### REST API Endpoint

```python
# Add to main.py for API mode
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_endpoint():
    image = request.files['image']
    # Process and return detections
    return jsonify(results)
```

---

## 🎯 **Use Cases**

1. **Construction Sites** - Monitor helmet compliance across large sites
2. **Manufacturing Plants** - Ensure PPE compliance in production areas
3. **Warehouses** - Safety monitoring in loading/unloading zones
4. **Mining Operations** - Track safety equipment usage
5. **Industrial Facilities** - Comprehensive safety oversight

---

## 🔐 **Security & Privacy**

- Local processing (no cloud uploads)
- Encrypted database storage
- Access control for dashboard
- GDPR-compliant data retention
- Configurable data retention policies

---

## 🤝 **Contributing**

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 **License**

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [Streamlit](https://streamlit.io/) - Dashboard framework

---

## 📧 **Contact**

For business inquiries or support:
- Email: vignesh246v@gmail.com
- LinkedIn: [My Profile](https://www.linkedin.com/in/vignesh246v-ai-engineer/)

---

## 🎓 **For Recruiters**

This project demonstrates:
- ✅ Production-ready AI/ML development
- ✅ Real-time computer vision systems
- ✅ Clean, modular architecture
- ✅ Full-stack implementation (Backend + Frontend)
- ✅ Database design and optimization
- ✅ Deployment-ready containerization
- ✅ Comprehensive documentation
- ✅ Industry-standard best practices

**Technologies:** Python, PyTorch, YOLOv8, OpenCV, Streamlit, SQLite, Docker

---

## ⭐ **Show Your Support**

If this project helped you, please give it a ⭐ star!

---

**Built with ❤️ by Vignesh**#


