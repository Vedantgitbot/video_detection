# Multimodal Deepfake Detection System

## Overview

This project implements a **multimodal deepfake detection pipeline** that combines multiple independent sources of evidence to classify a video as **Real** or **Fake**.

Instead of relying solely on visual artifacts, the system fuses:

* Facial motion dynamics
* Remote photoplethysmography (rPPG) features
* Audio-visual synchronization features
* Vision Transformer (ViT) semantic representations

These features are combined using an **XGBoost late-fusion classifier**, producing a final fake probability for each video.

---

## System Architecture

```
Input Video
     │
     ├─────────────────────────────────────────────┐
     │                                             │
     ▼                                             ▼
 Face Detection                              Audio Extraction
     │                                             │
     ▼                                             ▼
 Facial Landmarks                          Speech Features
     │                                             │
     ├───────────────┐                             │
     │               │                             │
     ▼               ▼                             ▼
 Motion Features   rPPG Features           Audio-Visual Sync
     │               │                             │
     └───────────────┴──────────────┐
                                    │
                                    ▼
                          Vision Transformer
                          (Pretrained ViT)
                                    │
                                    ▼
                      Logistic Regression Probe
                          (ViT Fake Score)
                                    │
                                    ▼
         Motion + rPPG + AV + ViT Feature Fusion
                                    │
                                    ▼
                          XGBoost Classifier
                                    │
                                    ▼
                      Fake Probability Prediction
                                    │
                                    ▼
                  Optional Platt Probability Calibration
                                    │
                                    ▼
                       Final Real / Fake Decision
```

---

## Features

### 1. Facial Motion Features
Extracted from facial landmarks — overall/mouth/jaw velocity, blink rate, Eye Aspect Ratio (EAR), motion jitter, and FFT motion energy.

### 2. rPPG Features
Remote Photoplethysmography estimates subtle skin color variation from blood flow — estimated heart rate and signal-to-noise ratio (SNR). Deepfakes often distort these physiological signals.

### 3. Audio-Visual Synchronization
Estimated lip-sync lag and audio-visual confidence, identifying inconsistencies between speech and lip movement.

### 4. Vision Transformer (ViT)
A pretrained ViT produces a semantic embedding for each video, reduced via a logistic regression probe into a single `vit_probe_score`.

---

## Feature Fusion

The project uses **late fusion**: each modality is processed independently, then combined into a final feature vector (motion + rPPG + audio-visual sync + ViT probe score) fed to the XGBoost classifier.

---

## Classifier

**XGBoost** was chosen for strong performance on tabular data, ability to capture nonlinear feature interactions, robustness on smaller datasets, and interpretable feature importance.

---

## Probability Calibration

Optional **Platt Scaling** improves probability calibration so confidence scores better reflect empirical probabilities. This affects probability estimates only, not model rankings.

---

## Evaluation

Evaluated using **Leave-One-Out Cross Validation (LOOCV)**, reporting Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, and Calibration Curves.

---

## Pipeline

1. Extract facial motion features
2. Extract rPPG features
3. Compute audio-visual synchronization features
4. Generate ViT embeddings + probe score
5. Fuse all features
6. Train XGBoost classifier
7. Predict fake probability
8. Optionally calibrate probabilities (Platt scaling)
9. Output final Real/Fake decision

---

## Repository Structure

```
video_detection/
│
├── Dataset/
│   ├── real/           # Place real videos here (not tracked in repo)
│   └── fake/           # Place fake videos here (not tracked in repo)
│
├── src/
│   ├── media.py         # Facial landmark & motion feature extraction
│   ├── rppg.py           # rPPG (heart rate / SNR) feature extraction
│   ├── vit.py             # ViT embedding + probe score
│   └── train.py           # Feature fusion, XGBoost training & evaluation
│
├── outputs/               # Generated at runtime (gitignored)
│   ├── features.csv
│   ├── vit_embeddings.csv
│   ├── calibration_curve.png
│   ├── calibration_curve_raw.png
│   ├── calibration_curve_platt.png
│   └── checkpoints/
│       └── model.joblib
│
├── .gitignore
└── README.md
```

> **Note:** `Dataset/real/` and `Dataset/fake/` ship as empty folders (with `.gitkeep`). Add your own `.mp4` videos to each before running the pipeline — video files are not tracked in this repo.

---

## Installation & Setup

```bash
conda create -n deepfake python=3.10 -y
conda activate deepfake
pip install -r requirements.txt
```

Audio-visual sync features require **FFmpeg**:
```bash
brew install ffmpeg
# OR
conda install -c conda-forge ffmpeg
ffmpeg -version   # verify
```

---

## Running the Project

```bash
python src/media.py     # facial motion features
python src/rppg.py       # rPPG features
python src/vit.py         # ViT embeddings + probe score
python src/train.py       # fuse features, train XGBoost, evaluate
```

---

## Current Performance

Evaluated on a balanced dataset of ~140 usable videos after feature merging:

* Accuracy ≈ 67%
* ROC-AUC ≈ 0.77
* F1 Score ≈ 0.69

A baseline demonstrating multimodal fusion can distinguish manipulated from authentic videos, with room for improvement via larger, more diverse datasets.

---

## Current Limitations

* Dataset size is still limited for robust generalization
* Handcrafted motion features may miss complex temporal deepfake artifacts
* Audio-sync features require FFmpeg to be installed and functioning correctly

---

## Future Work

* Larger, more diverse datasets (multiple languages, compression levels, deepfake generation methods)
* Additional physiological features (head pose, gaze tracking)
* Stronger temporal modeling (LSTMs, 3D CNNs, temporal ViTs)
* Cross-dataset evaluation
* Real-time inference optimization
* Robustness against unseen deepfake generation methods

---

## Technologies Used

Python · OpenCV · MediaPipe · NumPy · Pandas · Scikit-learn · XGBoost · PyTorch · Hugging Face Transformers · Vision Transformer (ViT) · FFmpeg

---

## Author

Vedant Brahmbhatt

## License

This project is intended for academic research and educational purposes.
