# Video Deepfake Detection Using Facial Motion Features

A machine learning-based video deepfake detection system that classifies videos as **real** or **fake** using facial motion patterns extracted from video frames.

The pipeline extracts facial landmarks, computes temporal motion features, optionally analyzes audio-video synchronization, and trains a classifier to detect manipulated videos.

---

## 📌 Project Overview

Deepfake videos often contain subtle inconsistencies in:
* **Facial movements** and micro-expressions
* **Mouth motion dynamics** and lip-sync anomalies
* **Jaw movement patterns** during speech
* **Eye blinking behavior** (frequency and duration)
* **Temporal landmark changes** across frames
* **Audio-video synchronization** lag

This project leverages these behavioral signals to distinguish between authentic and manipulated videos.

---

## ⚙️ System Pipeline

```
          [ Video Dataset ]
                 │
                 ▼
     [ Face Landmark Extraction ] (MediaPipe)
                 │
                 ▼
    [ Motion Feature Engineering ]
                 │
                 ▼
       [ Feature Dataset ] (CSV)
                 │
                 ▼
   [ Machine Learning Classifier ] (Scikit-Learn)
                 │
                 ▼
       [ Real / Fake Prediction ]
```

---

## 📊 Extracted Features

The system extracts facial motion-based features from each video, categorized into four core domains:

### 1. Facial Motion Features
| Feature | Description |
| :--- | :--- |
| `mouth_velocity_mean` | Average mouth movement speed |
| `mouth_velocity_std` | Variation in mouth movement |
| `jaw_velocity_mean` | Average jaw movement |
| `jaw_velocity_std` | Jaw movement variation |
| `overall_velocity_mean` | Overall facial landmark motion |
| `overall_velocity_std` | Temporal facial movement variation |

### 2. Eye & Blink Features
| Feature | Description |
| :--- | :--- |
| `blink_rate` | Estimated blinking frequency |
| `ear_mean` | Average Eye Aspect Ratio (EAR) |
| `ear_std` | Variation in Eye Aspect Ratio (EAR) |

### 3. Frequency Domain Features
Extracts motion frequency information to detect micro-jitters:
| Feature | Description |
| :--- | :--- |
| `mouth_jitter_fft_energy` | High-frequency mouth movement patterns |
| `jaw_jitter_fft_energy` | Jaw motion irregularities |
| `overall_jitter_fft_energy` | Overall facial jitter |

### 4. Audio-Video Features
The pipeline supports audio synchronization analysis:
| Feature | Description |
| :--- | :--- |
| `av_sync_lag_ms` | Audio-video timing difference |
| `av_sync_confidence` | Confidence of synchronization |

> ⚠️ **Note:** Audio synchronization features require FFmpeg to be installed on the system.

---

## 🚀 Installation & Setup

### Requirements
* Python 3.10+
* Conda (recommended)
* OpenCV
* MediaPipe
* Scikit-learn
* NumPy
* Pandas
* Joblib

### 1. Create & Activate Environment
```bash
conda create -n deepfake python=3.10 -y
conda activate deepfake
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg (Required for Audio-Video Sync)
#### macOS (via Homebrew or Conda)
```bash
brew install ffmpeg
# OR
conda install -c conda-forge ffmpeg
```

#### Verification
```bash
ffmpeg -version
```

---

## 📂 Project Structure

```text
video_detection/
│
├── Dataset/
│   ├── src/
│   │   ├── media.py       # Landmark & feature extraction script
│   │   └── train.py       # Classifier training & evaluation script
│   └── videos/            # Source videos organized by category
│
├── outputs/
│   ├── features.csv       # Extracted features table
│   └── checkpoints/
│       └── model.joblib   # Trained ML model
│
├── requirements.txt       # Dependencies list
└── README.md              # Project documentation
```

### Dataset Structure
The input videos should be organized as follows:
```text
Dataset/
├── real/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── fake/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

---

## 🏃 Running the Project

### Step 1: Extract Features
Execute the feature extraction script to process the videos and generate the structured dataset:
```bash
python Dataset/src/media.py
```

**Example Console Output:**
```text
Found 13 real videos, 11 fake videos

Processing [real] video.mp4 ...
Processing [fake] video.mp4 ...

Wrote 24 rows to outputs/features.csv
Generated file: outputs/features.csv
```

### Step 2: Train the Model
Run the training script to train the classifier and evaluate performance:
```bash
python Dataset/src/train.py
```

The script performs:
1. Feature loading & preprocessing
2. Label encoding (Real = 0, Fake = 1)
3. **Leave-One-Out Cross Validation (LOOCV)**
4. Model training
5. Feature importance analysis
6. Model serialization (`outputs/checkpoints/model.joblib`)

---

## 📈 Model Evaluation

### Evaluation Dataset
* **Total Videos:** 24
* **Real Videos:** 13
* **Fake Videos:** 11

### Performance Metrics (LOOCV)
* **Accuracy:** `0.625` (15 / 24 correct predictions)

#### Confusion Matrix
```text
              Predicted
              Real   Fake
Actual Real     8      5
Actual Fake     4      7
```

### Feature Importance
The relative impact of each facial motion metric on classification:
```text
mouth_velocity_mean          █████████████████████████ 44.8%
overall_velocity_std         ███████ 12.6%
jaw_velocity_std              █████ 9.1%
blink_rate                    ████ 7.9%
jaw_jitter_fft_energy         ███ 6.6%
overall_jitter_fft_energy     ███ 6.3%
```
*The model currently relies heavily on facial motion dynamics, particularly mouth movement patterns.*

---

## ⚠️ Current Limitations

1. **Small Dataset Size:** The experiment uses only 24 videos, which is insufficient for robust generalized detection. A minimum of **100+ videos** (ideally **500+**) is recommended.
2. **Audio Features Disabled (Optional):** Without FFmpeg, `av_sync_lag_ms` and `av_sync_confidence` default to `0`, preventing the detection of audio-video synchronicity discrepancies.
3. **Handcrafted Features:** Relying on engineered facial motion features may miss complex temporal deepfake artifacts.

---

## 🔮 Future Improvements

### 1. Dataset Expansion
* Add more real/fake samples to improve model generalization.
* Include diverse languages, environments, and compression levels.
* Add multiple deepfake generation methodologies (e.g., GANs, Diffusion, FaceSwap).

### 2. Feature Enhancements
Integrate advanced visual signals:
* **Head Pose Estimation** & Head Movement Jitter.
* **Eye Gaze Tracking** and pupil dynamics.
* **Facial Texture Artifacts** (chrominance loss, edge blurring).
* **Optical Flow Analysis** for temporal consistency.

### 3. Model Upgrades
* Transition from baseline classifiers to **XGBoost**, **LightGBM**, or **SVMs**.
* Explore Deep Learning architectures:
  * **CNN-based spatial features** (EfficientNet).
  * **Temporal Models** (LSTMs, GRUs, or 3D CNNs).
  * **Vision Transformers (ViTs)** for temporal attention.

---

## 🛠️ Technologies Used

* **Language:** Python 3.10+
* **Computer Vision:** OpenCV, MediaPipe Face Landmarker
* **Machine Learning:** Scikit-learn, NumPy, Pandas
* **Deep Learning Runtime:** TensorFlow Lite (MediaPipe dependency)
* **Multimedia Framework:** FFmpeg

---

## 👤 Author
* **Vedant Brahmbhatt**

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details. This repository is intended strictly for research and educational purposes.
