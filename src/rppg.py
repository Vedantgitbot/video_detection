"""
rppg.py
Extracts remote photoplethysmography (rPPG) features per video for deepfake
detection — a model-free, signal-processing-only feature. Real human skin
shows subtle, periodic color fluctuations from blood flow with each
heartbeat; this script measures how strong/periodic that pulse-like signal
is in a face region across a video. Some deepfake pipelines fail to
reproduce this biological signal convincingly.

Uses the CHROM method (de Haan & Jeanne, 2013): a chrominance-based
combination of R/G/B channel averages that's more robust to lighting and
motion than raw green-channel rPPG. No pretrained model, no fitting step —
these are deterministic signal-processing features, so there's no leakage
risk the way there is with the ViT probe.

Face ROI is a forehead sub-rectangle derived from the same face bounding box
used elsewhere in this pipeline (get_face_crop_bbox from media.py), rather
than precise individual landmark indices — kept simple and robust.

Uses VIDEO mode (dense, native-frame-rate sampling), NOT the sparse 1fps
sampling vit.py uses — rPPG needs a continuous, evenly-spaced signal at
native fps to resolve heart-rate frequencies (~0.7-4 Hz) without aliasing.

Writes its own CSV (outputs/rppg_features.csv) rather than modifying
features.csv or vit_embeddings.csv — train.py does the merging, same pattern
as the ViT branch.

Usage:
    python src/rppg.py --data_dir Dataset --out_csv outputs/rppg_features.csv

Requires: opencv-python, mediapipe, numpy, pandas, scipy (already installed
for media.py)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
except ImportError:
    print("mediapipe is required: pip install mediapipe opencv-python-headless", file=sys.stderr)
    raise

# Reuse model download/caching, video collection, and crop-bbox logic from
# media.py rather than duplicating it — keeps face detection identical across
# all three branches (motion, ViT, rPPG). Requires rppg.py to sit alongside
# media.py.
from media import (
    ensure_model_downloaded,
    collect_videos,
    get_face_crop_bbox,
    MAX_FRAMES_PER_VIDEO,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rppg")

MIN_FRAMES_REQUIRED = 60   # need a reasonably long window to resolve low
                            # heart-rate frequencies with any confidence

# --- Forehead ROI, expressed as fractions of the face bounding box ----------
# Deliberately a simple rectangular sub-region rather than precise landmark
# polygons — the forehead is a large, relatively flat, well-lit skin area
# that's easy to get approximately right and avoids relying on landmark
# indices that would need careful individual verification.
ROI_X_RANGE = (0.30, 0.70)   # horizontal: middle band, avoids temples/hairline edges
ROI_Y_RANGE = (0.05, 0.20)   # vertical: just below hairline, above eyebrows

# --- Plausible heart-rate band ----------------------------------------------
HR_MIN_BPM = 42.0
HR_MAX_BPM = 240.0


def extract_roi_mean_rgb(frame_bgr, bbox):
    """Given a face bbox (x1,y1,x2,y2), crops the forehead sub-rectangle and
    returns the mean (R, G, B) of that region, or None if the region is
    degenerate (too small / out of bounds)."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    rx1 = int(x1 + ROI_X_RANGE[0] * w)
    rx2 = int(x1 + ROI_X_RANGE[1] * w)
    ry1 = int(y1 + ROI_Y_RANGE[0] * h)
    ry2 = int(y1 + ROI_Y_RANGE[1] * h)

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    # BGR -> RGB order for readability downstream
    mean_bgr = roi.reshape(-1, 3).mean(axis=0)
    return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)


def chrom_signal(rgb_series: np.ndarray) -> np.ndarray:
    """
    CHROM method (de Haan & Jeanne, 2013): combines normalized R/G/B traces
    into a single chrominance-based pulse signal that's more robust to
    lighting changes and motion than raw green-channel rPPG.

    rgb_series: shape (n_frames, 3), columns = [R, G, B] means per frame.
    Returns a 1D pulse signal, or an empty array if the input is degenerate.
    """
    if len(rgb_series) < 4:
        return np.array([])

    means = rgb_series.mean(axis=0)
    if np.any(means < 1e-6):
        return np.array([])

    # Normalize each channel by its own mean (removes DC / overall brightness)
    normalized = rgb_series / means

    r_n, g_n, b_n = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    x = 3 * r_n - 2 * g_n
    y = 1.5 * r_n + g_n - 1.5 * b_n

    std_x, std_y = np.std(x), np.std(y)
    if std_y < 1e-8:
        return x - np.mean(x)  # degenerate fallback, shouldn't normally happen

    alpha = std_x / std_y
    signal = x - alpha * y
    return signal - np.mean(signal)


def bandpass_filter(signal: np.ndarray, fps: float, low_bpm: float, high_bpm: float):
    """Butterworth bandpass filter restricting the signal to plausible
    heart-rate frequencies. Returns None if the signal is too short or fps
    doesn't allow the requested band (e.g. very low-fps video)."""
    if len(signal) < 8:
        return None

    nyquist = fps / 2.0
    low_hz = (low_bpm / 60.0) / nyquist
    high_hz = (high_bpm / 60.0) / nyquist

    # Guard against invalid filter bands (e.g. extremely low source fps)
    low_hz = max(low_hz, 1e-4)
    high_hz = min(high_hz, 0.999)
    if low_hz >= high_hz:
        return None

    try:
        b, a = butter(N=3, Wn=[low_hz, high_hz], btype="band")
        return filtfilt(b, a, signal)
    except ValueError:
        return None


def compute_rppg_features(rgb_series: np.ndarray, fps: float) -> dict:
    """
    Runs CHROM + bandpass filtering + spectral analysis on a video's forehead
    RGB trace. Returns:
      rppg_snr: ratio of spectral energy near the detected peak frequency to
                total energy in the heart-rate band — higher means a
                cleaner, more periodic (more "biologically real") pulse
                signal. 0.0 if no usable signal could be extracted.
      rppg_peak_bpm: the dominant frequency in the heart-rate band, in BPM.
                0.0 if no usable signal.
    """
    default = {"rppg_snr": 0.0, "rppg_peak_bpm": 0.0}

    pulse = chrom_signal(rgb_series)
    if pulse.size == 0:
        return default

    filtered = bandpass_filter(pulse, fps, HR_MIN_BPM, HR_MAX_BPM)
    if filtered is None or np.std(filtered) < 1e-8:
        return default

    spectrum = np.abs(np.fft.rfft(filtered)) ** 2
    freqs_hz = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
    freqs_bpm = freqs_hz * 60.0

    band_mask = (freqs_bpm >= HR_MIN_BPM) & (freqs_bpm <= HR_MAX_BPM)
    if not np.any(band_mask):
        return default

    band_spectrum = spectrum[band_mask]
    band_freqs = freqs_bpm[band_mask]
    total_band_energy = band_spectrum.sum()
    if total_band_energy < 1e-12:
        return default

    peak_idx = int(np.argmax(band_spectrum))
    peak_bpm = float(band_freqs[peak_idx])

    # "SNR": energy within a small window around the peak vs total band
    # energy — a real, strong periodic pulse concentrates energy narrowly
    # around one frequency; noise spreads energy more evenly across the band.
    window = max(1, len(band_spectrum) // 20)  # ~5% of band width on each side
    lo = max(0, peak_idx - window)
    hi = min(len(band_spectrum), peak_idx + window + 1)
    peak_energy = band_spectrum[lo:hi].sum()
    snr = float(peak_energy / total_band_energy)

    return {"rppg_snr": snr, "rppg_peak_bpm": peak_bpm}


def extract_video_rppg(video_path: Path, model_path: Path):
    """
    Runs MediaPipe FaceLandmarker (VIDEO mode, dense per-frame — same
    cadence as media.py, unlike vit.py's sparse 1fps sampling) to get a face
    bbox per frame, extracts the forehead ROI's mean RGB, and computes
    rPPG features from the resulting per-frame RGB trace.

    Returns a dict of features, or None if too few frames had a usable face.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Could not open video (corrupt or unsupported): {video_path.name}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0
    frame_interval_ms = 1000.0 / fps

    options = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rgb_frames = []
    frame_count = 0

    try:
        with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                if frame_count >= MAX_FRAMES_PER_VIDEO:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except cv2.error:
                    frame_count += 1
                    continue

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(frame_count * frame_interval_ms)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                frame_count += 1

                if not result.face_landmarks:
                    continue

                lm = result.face_landmarks[0]
                h, w = frame.shape[:2]
                bbox = get_face_crop_bbox(lm, w, h, margin=0.0)  # tight bbox, no padding, for a stable ROI

                mean_rgb = extract_roi_mean_rgb(frame, bbox)
                if mean_rgb is not None:
                    rgb_frames.append(mean_rgb)
    finally:
        cap.release()

    if len(rgb_frames) < MIN_FRAMES_REQUIRED:
        log.warning(
            f"{video_path.name}: only {len(rgb_frames)} usable forehead-ROI frames "
            f"(need >= {MIN_FRAMES_REQUIRED}), skipping rPPG for this video"
        )
        return None

    rgb_series = np.stack(rgb_frames)
    return compute_rppg_features(rgb_series, fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Dataset")
    parser.add_argument("--out_csv", default="outputs/rppg_features.csv")
    parser.add_argument("--resume", action="store_true",
                         help="Skip videos already present in an existing out_csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        log.error(f"data_dir does not exist: {data_dir}")
        sys.exit(1)

    real_videos = collect_videos(data_dir, "real")
    fake_videos = collect_videos(data_dir, "fake")
    log.info(f"Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")

    if len(real_videos) == 0 or len(fake_videos) == 0:
        log.error("Need at least one video in both real/ and fake/ folders.")
        sys.exit(1)

    try:
        model_path = ensure_model_downloaded()
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    out_path = Path(args.out_csv)
    already_done = set()
    rows = []
    if args.resume and out_path.exists():
        existing = pd.read_csv(out_path)
        already_done = set(existing["filename"].tolist())
        rows.extend(existing.to_dict("records"))
        log.info(f"Resuming: {len(already_done)} videos already in {out_path}, skipping them")

    start_time = time.time()

    for label, video_list in [("real", real_videos), ("fake", fake_videos)]:
        for video_path in video_list:
            if video_path.name in already_done:
                continue

            log.info(f"Processing [{label}] {video_path.name}")
            try:
                feats = extract_video_rppg(video_path, model_path)
            except Exception as e:
                log.error(f"Unexpected error on {video_path.name}: {e}")
                feats = None

            if feats is None:
                continue

            row = {"filename": video_path.name, "label": label}
            row.update(feats)
            rows.append(row)

            # Write incrementally so a crash/interrupt doesn't lose progress.
            pd.DataFrame(rows).to_csv(out_path, index=False)

    if not rows:
        log.error("No videos produced usable rPPG features. Nothing to write.")
        sys.exit(1)

    elapsed = time.time() - start_time
    log.info(f"Wrote {len(rows)} rows to {out_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()