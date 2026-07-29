"""
vit.py
Extracts per-video appearance embeddings using a pretrained Vision Transformer
(ViT), for fusion with the motion features from media.py.

This script only produces the *raw averaged embedding* per video. The
dimensionality reduction (a small logistic-regression "probe" fit fold-by-fold
inside nested LOOCV) happens in train.py, so it never sees test data during
fitting.

Face crops are driven off the same MediaPipe FaceLandmarker detection used in
media.py (via get_face_crop_bbox), so the appearance branch sees the same face
region as the motion branch rather than a separately-tuned crop.

Samples ~1 frame/sec (not every frame) since appearance embeddings don't need
frame-level temporal resolution the way motion features do, which keeps
runtime reasonable.

Usage:
    python src/vit.py --data_dir Dataset --out_csv outputs/vit_embeddings.csv

Requires: torch, transformers, pillow
    pip install torch transformers pillow
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
except ImportError:
    print("mediapipe is required: pip install mediapipe opencv-python-headless", file=sys.stderr)
    raise

try:
    import torch
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    print("torch and transformers are required: pip install torch transformers", file=sys.stderr)
    raise

# Reuse the model download/caching, video collection, and crop-bbox logic from
# media.py rather than duplicating it — keeps face detection identical across
# the motion and appearance branches. Requires vit.py to sit alongside media.py.
from media import (
    ensure_model_downloaded,
    collect_videos,
    get_face_crop_bbox,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("vit")

# Small, fast ViT-style backbone that runs comfortably on an M-series Mac via
# the MPS backend. DINOv2-small tends to produce more useful features for
# fine-grained visual tasks than a generic ImageNet-classification ViT.
DEFAULT_MODEL_NAME = "facebook/dinov2-small"

SAMPLE_FPS = 1.0            # how densely to sample frames for the ViT pass
MAX_SAMPLED_FRAMES = 60     # cap: 60 sampled frames = 60s of video at 1fps
MIN_SAMPLED_FRAMES = 3      # too few face-detected frames -> skip video


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_vit(model_name: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return processor, model


@torch.no_grad()
def embed_crop(crop_bgr: np.ndarray, processor, model, device: str) -> np.ndarray:
    """Runs a single face-crop image through the ViT and returns a 1D embedding
    (mean-pooled over patch tokens, which tends to be more stable than the CLS
    token alone for this kind of transfer-feature use)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # [1, tokens, dim] -> [dim]
    return embedding.cpu().numpy().astype(np.float32)


def extract_video_embedding(video_path: Path, model_path: Path, processor, model, device: str):
    """Samples ~SAMPLE_FPS frames per second, crops the face using MediaPipe
    landmarks, embeds each crop with the ViT, and averages across frames.
    Returns a 1D numpy array, or None if too few frames had a usable face."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Could not open video (corrupt or unsupported): {video_path.name}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0
    frame_stride = max(1, int(round(fps / SAMPLE_FPS)))

    options = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        # IMAGE mode, not VIDEO: we're sparsely sampling non-contiguous frames,
        # so there's no need for (and VIDEO mode would reject) the strict
        # per-frame timestamp tracking that media.py uses.
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )

    embeddings = []
    frame_idx = 0
    sampled = 0

    try:
        with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                if sampled >= MAX_SAMPLED_FRAMES:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % frame_stride != 0:
                    frame_idx += 1
                    continue

                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except cv2.error:
                    frame_idx += 1
                    continue

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)
                frame_idx += 1
                sampled += 1

                if not result.face_landmarks:
                    continue

                lm = result.face_landmarks[0]
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = get_face_crop_bbox(lm, w, h)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                try:
                    emb = embed_crop(crop, processor, model, device)
                except Exception as e:
                    log.warning(f"{video_path.name}: ViT embedding failed on a frame: {e}")
                    continue

                embeddings.append(emb)
    finally:
        cap.release()

    if len(embeddings) < MIN_SAMPLED_FRAMES:
        log.warning(
            f"{video_path.name}: only {len(embeddings)} usable face crops "
            f"(need >= {MIN_SAMPLED_FRAMES}), skipping"
        )
        return None

    return np.mean(np.stack(embeddings), axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Dataset")
    parser.add_argument("--out_csv", default="outputs/vit_embeddings.csv")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME,
                         help="HuggingFace model name/path for the ViT backbone")
    parser.add_argument("--resume", action="store_true",
                         help="Skip videos already present in an existing out_csv "
                              "(useful if a long run gets interrupted)")
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

    device = get_device()
    log.info(f"Using device: {device}")
    log.info(f"Loading ViT backbone: {args.model_name} (first run downloads weights)")
    processor, vit_model = load_vit(args.model_name, device)

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
                emb = extract_video_embedding(video_path, model_path, processor, vit_model, device)
            except Exception as e:
                log.error(f"Unexpected error on {video_path.name}: {e}")
                emb = None

            if emb is None:
                continue

            row = {"filename": video_path.name, "label": label}
            row.update({f"vit_{i}": float(v) for i, v in enumerate(emb)})
            rows.append(row)

            # Write incrementally so a crash/interrupt partway through a long
            # run doesn't lose everything already computed.
            pd.DataFrame(rows).to_csv(out_path, index=False)

    if not rows:
        log.error("No videos produced usable embeddings. Nothing to write.")
        sys.exit(1)

    elapsed = time.time() - start_time
    log.info(f"Wrote {len(rows)} rows to {out_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()