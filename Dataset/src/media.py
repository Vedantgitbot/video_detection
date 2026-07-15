"""
media.py
Extracts per-video facial motion features for deepfake detection.

Expects a data directory structured as:
    data_dir/
        real/*.mp4 (.mov, .avi, .mkv)
        fake/*.mp4 (.mov, .avi, .mkv)

Writes outputs/features.csv with columns matching train.py's FEATURE_COLS:
    filename, label,
    blink_rate, ear_mean, ear_std,
    jaw_velocity_mean, jaw_velocity_std, jaw_jitter_fft_energy,
    mouth_velocity_mean, mouth_velocity_std, mouth_jitter_fft_energy,
    overall_velocity_mean, overall_velocity_std, overall_jitter_fft_energy,
    av_sync_lag_ms, av_sync_confidence

Requires ffmpeg on PATH for the audio-visual sync features (used to extract
the audio track). If ffmpeg is missing or a clip has no audio, those two
columns fall back to 0.0 rather than failing the whole video.

Usage:
    python src/media.py --data_dir Dataset --out_csv outputs/features.csv
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import correlate

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
except ImportError:
    print("mediapipe is required: pip install mediapipe opencv-python-headless", file=sys.stderr)
    raise

# Official Google-hosted MediaPipe face landmarker model (current Tasks API).
# Only ever fetched from this specific, hardcoded HTTPS URL — never a
# user-supplied or dynamically constructed one — to avoid pulling an
# arbitrary/untrusted model file.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_DIR = Path.home() / ".cache" / "mediapipe_models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"
MODEL_MIN_BYTES = 500_000   # sanity floor: a truncated/corrupt download would be tiny
MODEL_MAX_BYTES = 50_000_000  # sanity ceiling: this model is a few MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("media")


def ensure_model_downloaded() -> Path:
    """Downloads the face landmarker model once and caches it locally.
    Validates size bounds so a truncated or unexpectedly huge response
    (e.g. an error page instead of the model) doesn't get silently used."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MODEL_MIN_BYTES:
        return MODEL_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading face landmarker model from {MODEL_URL} ...")
    tmp_path = MODEL_PATH.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Unexpected HTTP status {resp.status}")
            data = resp.read(MODEL_MAX_BYTES + 1)
        if len(data) < MODEL_MIN_BYTES:
            raise RuntimeError(f"Downloaded file too small ({len(data)} bytes) - likely corrupt")
        if len(data) > MODEL_MAX_BYTES:
            raise RuntimeError("Downloaded file larger than expected - refusing to use")
        tmp_path.write_bytes(data)
        tmp_path.rename(MODEL_PATH)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Could not download required model file: {e}. "
            f"Check your internet connection, or manually place the model at {MODEL_PATH}"
        ) from e

    log.info(f"Model cached at {MODEL_PATH}")
    return MODEL_PATH


# --- Security / resource-limit constants -----------------------------------
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FRAMES_PER_VIDEO = 1800     # ~60s at 30fps hard cap: bounds memory/CPU per file
MIN_FRAMES_REQUIRED = 10        # too few frames -> features are meaningless, skip
FFMPEG_TIMEOUT_SEC = 60         # bound how long audio extraction can take per file

# --- MediaPipe FaceMesh landmark indices (standard 6-pt EAR formulation) ---
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
CHIN = 152
MOUTH_UPPER = 13
MOUTH_LOWER = 14
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
# Sparse but stable subset across the face, used for "overall" motion signal
OVERALL_SUBSET = [1, 33, 61, 152, 263, 291, 199, 4]

EAR_BLINK_THRESHOLD = 0.21
BLINK_REFRACTORY_FRAMES = 3  # minimum frames between counted blinks (avoid double count)

# --- Audio-visual sync constants --------------------------------------------
AV_SYNC_MAX_LAG_MS = 500.0   # search window: real lip-sync offsets are well under this
AV_AUDIO_SAMPLE_RATE = 16000


def eye_aspect_ratio(landmarks, eye_idx):
    """Standard EAR: vertical eye distances over horizontal eye distance."""
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_idx]
    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal < 1e-8:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def fft_jitter_energy(signal, high_freq_fraction=0.5):
    """
    Ratio of energy in the upper half of the frequency spectrum to total energy.
    High-frequency-dominant motion signals indicate frame-to-frame instability
    (jitter) rather than smooth physiological movement.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 4 or np.all(signal == signal[0]):
        return 0.0
    signal = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    if spectrum.sum() < 1e-12:
        return 0.0
    cutoff = int(len(spectrum) * (1 - high_freq_fraction))
    high_energy = spectrum[cutoff:].sum()
    total_energy = spectrum.sum()
    return float(high_energy / total_energy)


def extract_audio_envelope(video_path: Path, num_frames: int, frame_interval_ms: float):
    """
    Extracts the audio track via ffmpeg and computes an RMS energy envelope,
    one value per video frame interval, so it lines up 1:1 with the per-frame
    mouth-motion signal for cross-correlation.

    Returns None (not an error) if the video has no audio track, ffmpeg is
    unavailable, or extraction otherwise fails — callers must treat that as
    "sync features unavailable for this video" rather than a hard failure.
    """
    if num_frames < 4 or frame_interval_ms <= 0:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(AV_AUDIO_SAMPLE_RATE),
            "-f", "wav", str(wav_path),
        ]
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=FFMPEG_TIMEOUT_SEC,
            )
        except FileNotFoundError:
            log.warning("ffmpeg not found on PATH — skipping audio sync features")
            return None
        except subprocess.TimeoutExpired:
            log.warning(f"{video_path.name}: ffmpeg audio extraction timed out")
            return None

        if result.returncode != 0 or not wav_path.exists():
            log.info(f"{video_path.name}: no audio track extracted (silent clip or extraction failed)")
            return None

        try:
            sr, samples = wavfile.read(wav_path)
        except Exception as e:
            log.warning(f"{video_path.name}: could not read extracted audio: {e}")
            return None

    if samples.size == 0:
        return None
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float64)

    window_size = int(sr * frame_interval_ms / 1000.0)
    if window_size < 1:
        return None

    envelope = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * window_size
        end = start + window_size
        chunk = samples[start:end]
        envelope[i] = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0

    if np.std(envelope) < 1e-8:
        return None  # flat/silent envelope carries no sync information

    return envelope


def compute_av_sync_features(mouth_v: np.ndarray, audio_envelope, fps: float) -> dict:
    """
    Cross-correlates mouth-opening velocity against the audio energy envelope
    to detect audio-visual sync offset — generated video/audio pipelines are
    often only loosely coupled in time, so a consistent nonzero lag or a very
    low peak correlation can be a deepfake signal.

    Returns:
      av_sync_lag_ms: offset (ms) at peak correlation. Positive = audio leads
                       mouth motion. Near-zero for well-synced real speech.
      av_sync_confidence: normalized peak correlation strength, 0-1. Low
                       values mean weak audio-visual coupling overall — note
                       this can also just mean a quiet/non-talking clip, not
                       only synthesis, so treat it as one signal among several.
    """
    default = {"av_sync_lag_ms": 0.0, "av_sync_confidence": 0.0}
    if audio_envelope is None or len(audio_envelope) < 4 or len(mouth_v) < 4:
        return default

    n = min(len(mouth_v), len(audio_envelope))
    m = mouth_v[:n] - np.mean(mouth_v[:n])
    a = audio_envelope[:n] - np.mean(audio_envelope[:n])

    if np.std(m) < 1e-8 or np.std(a) < 1e-8:
        return default

    m_norm = m / np.std(m)
    a_norm = a / np.std(a)

    max_lag_frames = max(1, min(int(AV_SYNC_MAX_LAG_MS / 1000.0 * fps), n - 1))

    full_corr = correlate(a_norm, m_norm, mode="full")
    lags = np.arange(-(len(m_norm) - 1), len(a_norm))
    center = len(full_corr) // 2
    lo = max(0, center - max_lag_frames)
    hi = min(len(full_corr), center + max_lag_frames + 1)

    windowed_corr = full_corr[lo:hi]
    windowed_lags = lags[lo:hi]
    if len(windowed_corr) == 0:
        return default

    windowed_corr = windowed_corr / n  # normalized cross-correlation, ~[-1, 1] scale given unit-variance inputs

    best_idx = int(np.argmax(np.abs(windowed_corr)))
    best_lag_frames = int(windowed_lags[best_idx])
    best_corr = float(windowed_corr[best_idx])

    return {
        "av_sync_lag_ms": float(best_lag_frames / fps * 1000.0),
        "av_sync_confidence": float(min(abs(best_corr), 1.0)),
    }


def safe_resolve_video_path(base_dir: Path, candidate: Path) -> Path:
    """
    Resolves candidate path and verifies it is actually inside base_dir.
    Prevents path traversal (e.g. a crafted filename like '../../etc/passwd')
    from causing reads outside the intended dataset directory.
    """
    base_resolved = base_dir.resolve()
    candidate_resolved = candidate.resolve()
    if base_resolved not in candidate_resolved.parents and candidate_resolved != base_resolved:
        raise ValueError(f"Path escapes data_dir, refusing to process: {candidate}")
    return candidate_resolved


def extract_video_features(video_path: Path, model_path: Path):
    """
    Runs MediaPipe FaceLandmarker (Tasks API, VIDEO mode) over a video and
    returns aggregated motion + audio-visual sync features, or None if the
    video is unreadable / has no usable face detections.

    A fresh landmarker is created per video (rather than reused across videos)
    because VIDEO mode requires strictly increasing timestamps; reusing one
    instance across files risks timestamp/state leakage between videos.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Could not open video (corrupt or unsupported): {video_path.name}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0  # sane fallback; some containers report garbage fps
    frame_interval_ms = 1000.0 / fps

    ear_series = []
    jaw_pos, mouth_pos, overall_pos = [], [], []
    frame_count = 0

    options = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                if frame_count >= MAX_FRAMES_PER_VIDEO:
                    log.info(f"{video_path.name}: hit frame cap ({MAX_FRAMES_PER_VIDEO}), truncating")
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except cv2.error:
                    frame_count += 1
                    continue  # corrupt frame, skip rather than crash whole video

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(frame_count * frame_interval_ms)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                frame_count += 1

                if not result.face_landmarks:
                    continue

                lm = result.face_landmarks[0]

                # Normalize distances by inter-ocular distance so features are
                # comparable across faces at different distances/resolutions.
                l_outer = np.array([lm[LEFT_EYE_OUTER].x, lm[LEFT_EYE_OUTER].y])
                r_outer = np.array([lm[RIGHT_EYE_OUTER].x, lm[RIGHT_EYE_OUTER].y])
                iod = np.linalg.norm(l_outer - r_outer)
                if iod < 1e-6:
                    continue

                ear = (eye_aspect_ratio(lm, RIGHT_EYE) + eye_aspect_ratio(lm, LEFT_EYE)) / 2.0
                ear_series.append(ear)

                jaw_pos.append(np.array([lm[CHIN].x, lm[CHIN].y]) / iod)
                mouth_pos.append(
                    np.array([
                        (lm[MOUTH_UPPER].x + lm[MOUTH_LOWER].x) / 2.0,
                        (lm[MOUTH_UPPER].y + lm[MOUTH_LOWER].y) / 2.0,
                    ]) / iod
                )
                overall_pos.append(
                    np.mean([[lm[i].x, lm[i].y] for i in OVERALL_SUBSET], axis=0) / iod
                )
    finally:
        cap.release()

    if len(ear_series) < MIN_FRAMES_REQUIRED:
        log.warning(
            f"{video_path.name}: only {len(ear_series)} usable frames "
            f"(face not detected reliably), skipping"
        )
        return None

    def velocity_series(pos_list):
        pos = np.array(pos_list)
        diffs = np.diff(pos, axis=0)
        speed = np.linalg.norm(diffs, axis=1) * fps  # units/sec, normalized coords
        return speed

    jaw_v = velocity_series(jaw_pos)
    mouth_v = velocity_series(mouth_pos)
    overall_v = velocity_series(overall_pos)

    # Blink counting: threshold crossing with refractory period
    ear_arr = np.array(ear_series)
    blinks = 0
    i = 0
    while i < len(ear_arr):
        if ear_arr[i] < EAR_BLINK_THRESHOLD:
            blinks += 1
            i += BLINK_REFRACTORY_FRAMES
        else:
            i += 1
    duration_sec = frame_count / fps if fps > 0 else 1.0
    blink_rate = blinks / duration_sec if duration_sec > 0 else 0.0

    # --- Audio-visual sync features ---
    # Aligned against mouth_v (mouth-opening velocity), which is one shorter
    # than mouth_pos due to np.diff; audio envelope is extracted at the same
    # per-frame cadence and trimmed to match inside compute_av_sync_features.
    audio_envelope = extract_audio_envelope(
        video_path, num_frames=len(mouth_pos), frame_interval_ms=frame_interval_ms
    )
    av_features = compute_av_sync_features(mouth_v, audio_envelope, fps)

    features = {
        "blink_rate": blink_rate,
        "ear_mean": float(np.mean(ear_arr)),
        "ear_std": float(np.std(ear_arr)),
        "jaw_velocity_mean": float(np.mean(jaw_v)) if len(jaw_v) else 0.0,
        "jaw_velocity_std": float(np.std(jaw_v)) if len(jaw_v) else 0.0,
        "jaw_jitter_fft_energy": fft_jitter_energy(jaw_v),
        "mouth_velocity_mean": float(np.mean(mouth_v)) if len(mouth_v) else 0.0,
        "mouth_velocity_std": float(np.std(mouth_v)) if len(mouth_v) else 0.0,
        "mouth_jitter_fft_energy": fft_jitter_energy(mouth_v),
        "overall_velocity_mean": float(np.mean(overall_v)) if len(overall_v) else 0.0,
        "overall_velocity_std": float(np.std(overall_v)) if len(overall_v) else 0.0,
        "overall_jitter_fft_energy": fft_jitter_energy(overall_v),
        "av_sync_lag_ms": av_features["av_sync_lag_ms"],
        "av_sync_confidence": av_features["av_sync_confidence"],
    }
    return features


def collect_videos(data_dir: Path, label: str):
    label_dir = data_dir / label
    if not label_dir.is_dir():
        log.warning(f"Expected folder not found: {label_dir}")
        return []
    videos = []
    for p in sorted(label_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
            try:
                videos.append(safe_resolve_video_path(data_dir, p))
            except ValueError as e:
                log.error(str(e))
    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Dataset",
                         help="Directory containing real/ and fake/ subfolders of videos")
    parser.add_argument("--out_csv", default="outputs/features.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        log.error(f"data_dir does not exist: {data_dir}")
        sys.exit(1)

    real_videos = collect_videos(data_dir, "real")
    fake_videos = collect_videos(data_dir, "fake")
    log.info(f"Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")

    if len(real_videos) == 0 or len(fake_videos) == 0:
        log.error(
            "Need at least one video in both real/ and fake/ folders. "
            f"Expected structure: {data_dir}/real/*.mp4 and {data_dir}/fake/*.mp4"
        )
        sys.exit(1)

    try:
        model_path = ensure_model_downloaded()
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    rows = []
    start_time = time.time()

    for label, video_list in [("real", real_videos), ("fake", fake_videos)]:
        for video_path in video_list:
            log.info(f"Processing [{label}] {video_path.name}")
            try:
                feats = extract_video_features(video_path, model_path)
            except Exception as e:
                # Never let one bad file kill the whole batch.
                log.error(f"Unexpected error on {video_path.name}: {e}")
                feats = None

            if feats is None:
                continue

            feats["filename"] = video_path.name
            feats["label"] = label
            rows.append(feats)

    if not rows:
        log.error("No videos produced usable features. Nothing to write.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    col_order = ["filename", "label"] + [c for c in df.columns if c not in ("filename", "label")]
    df = df[col_order]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    elapsed = time.time() - start_time
    log.info(f"Wrote {len(df)} rows to {out_path} in {elapsed:.1f}s")
    log.info(f"Label counts:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    main()