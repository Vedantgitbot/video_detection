"""
train.py
Loads outputs/features.csv (+ optional outputs/rppg_features.csv and
outputs/vit_embeddings.csv), trains an XGBoost classifier, reports LOOCV
accuracy plus a full evaluation suite: ROC-AUC, precision/recall/F1, and a
calibration curve (reliability diagram) — both RAW and PLATT-CALIBRATED.

VIT PROBE MODE COMPARISON (--vit_mode pca | direct):
  pca (default, previous behavior): 384-dim ViT embedding -> fold-safe PCA
    (--pca_components, default 10) -> strongly-regularized logistic
    regression (C=0.1) on the reduced embedding.
  direct: skips PCA entirely. Fits a MORE strongly-regularized logistic
    regression (C=0.01 by default, override with --vit_c) directly on the
    scaled 384-dim embedding. Same fold-safe discipline throughout (inner
    K-fold for training-row out-of-fold scores, full-fold fit for the
    held-out test score) — only the dimensionality-reduction step differs.

  This is a genuine, open comparison: PCA is unsupervised and may discard
  label-relevant variance; direct fitting sees the full embedding but needs
  much stronger regularization to avoid overfitting ~110-125 training rows
  against 384 features. Run both and compare vit_probe_score's contribution
  and overall LOOCV metrics to see which actually helps.

CALIBRATION (Platt scaling), fold-safe: see prior version's docstring —
unchanged. Isotonic was deliberately not used (needs more data than ~140
points to avoid overfitting the calibration mapping itself).

Usage:
    # current default (PCA probe)
    python src/train.py --vit_mode pca --pca_components 10

    # new: direct probe on raw 384-dim embedding
    python src/train.py --vit_mode direct --vit_c 0.01
"""

import argparse
import os

import numpy as np
np.random.seed(42)
import pandas as pd
import joblib
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score,
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


FEATURE_COLS = [
    "blink_rate", "ear_mean", "ear_std",
    "jaw_velocity_mean", "jaw_velocity_std", "jaw_jitter_fft_energy",
    "mouth_velocity_mean", "mouth_velocity_std", "mouth_jitter_fft_energy",
    "overall_velocity_mean", "overall_velocity_std", "overall_jitter_fft_energy",
    "brow_raise_mean", "head_yaw_std",
    "av_sync_lag_ms", "av_sync_confidence",
]

VIT_PROBE_COL = "vit_probe_score"
RPPG_COLS = ["rppg_snr", "rppg_peak_bpm"]
INNER_KFOLD_SPLITS = 5
DEFAULT_PCA_COMPONENTS = 10
DEFAULT_CALIBRATION_BINS = 5
DECISION_THRESHOLD = 0.5

# Regularization strength defaults per ViT probe mode. 'direct' needs much
# stronger L2 than 'pca' since it's fitting ~384 coefficients against
# ~110-125 training rows per fold, versus ~10 coefficients in PCA mode.
VIT_C_BY_MODE = {"pca": 0.1, "direct": 0.01}


def fit_vit_pipeline(vit_train, y_train, n_components, mode="pca", vit_c=None):
    """Fits StandardScaler -> [PCA if mode='pca'] -> strongly-regularized
    logistic regression on ViT embeddings. Returns (scaler, pca_or_None, probe).

    mode='pca': reduces to n_components (fold-safe, capped by fold size)
                before the probe — the original approach.
    mode='direct': skips PCA, fits the probe directly on the scaled 384-dim
                embedding with stronger regularization (see VIT_C_BY_MODE).
    """
    if vit_c is None:
        vit_c = VIT_C_BY_MODE[mode]

    scaler = StandardScaler().fit(vit_train)
    scaled = scaler.transform(vit_train)

    if mode == "pca":
        max_components = min(n_components, scaled.shape[0] - 1, scaled.shape[1])
        max_components = max(1, max_components)
        pca = PCA(n_components=max_components, random_state=42).fit(scaled)
        probe_input = pca.transform(scaled)
    elif mode == "direct":
        pca = None
        probe_input = scaled
    else:
        raise ValueError(f"Unknown vit_mode: {mode}")

    probe = LogisticRegression(C=vit_c, max_iter=2000, random_state=42)
    probe.fit(probe_input, y_train)
    return scaler, pca, probe


def vit_pipeline_scores(scaler, pca, probe, vit_rows):
    scaled = scaler.transform(vit_rows)
    probe_input = pca.transform(scaled) if pca is not None else scaled
    return probe.predict_proba(probe_input)[:, 1]


def fold_safe_train_scores(vit_train, y_train, n_components, mode="pca", vit_c=None,
                            n_splits=INNER_KFOLD_SPLITS, random_state=42):
    n = len(y_train)
    k = min(n_splits, n)
    if k < 2:
        scaler, pca, probe = fit_vit_pipeline(vit_train, y_train, n_components, mode, vit_c)
        return vit_pipeline_scores(scaler, pca, probe, vit_train)
    scores = np.zeros(n)
    inner_kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    for inner_train_idx, inner_val_idx in inner_kf.split(vit_train):
        scaler, pca, probe = fit_vit_pipeline(
            vit_train[inner_train_idx], y_train[inner_train_idx], n_components, mode, vit_c
        )
        scores[inner_val_idx] = vit_pipeline_scores(
            scaler, pca, probe, vit_train[inner_val_idx]
        )
    return scores


def fit_platt_scaler(raw_scores, y_train):
    platt = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    platt.fit(raw_scores.reshape(-1, 1), y_train)
    return platt


def platt_transform(platt, raw_scores):
    return platt.predict_proba(np.asarray(raw_scores).reshape(-1, 1))[:, 1]


def print_calibration_curve(y_true, y_proba, n_bins, plot_path=None, label=""):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_proba, bin_edges[1:-1])
    counts = np.array([(bin_idx == i).sum() for i in range(n_bins)])
    nonempty = counts > 0

    print(f"\nCalibration curve [{label}] ({n_bins} bins, out-of-fold LOOCV probabilities):")
    print(f"  {'bin range':<14}{'n videos':<10}{'avg predicted':<16}{'actual fake %':<14}")
    edges_nonempty = bin_edges[:-1][nonempty]
    for lo, mp, fp, n in zip(edges_nonempty, mean_predicted, fraction_pos, counts[nonempty]):
        hi = lo + (1.0 / n_bins)
        print(f"  [{lo:.2f}-{hi:.2f})  {n:<10}{mp:<16.3f}{fp:<14.3f}")
    print("  (well-calibrated = 'avg predicted' close to 'actual fake %' in every row)")

    if plot_path is not None and HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
        ax.plot(mean_predicted, fraction_pos, marker="o", label=f"model ({label})")
        ax.set_xlabel("Mean predicted probability (fake)")
        ax.set_ylabel("Actual fraction fake")
        ax.set_title(f"Calibration curve — {label}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Calibration plot saved to {plot_path}")
    elif plot_path is not None and not HAS_MATPLOTLIB:
        print("  (matplotlib not installed — skipped saving plot, text table above still valid)")


def print_eval_metrics(y_true, y_proba, threshold, label=""):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = float("nan")
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    print(f"\nEvaluation metrics [{label}] (positive class = fake, threshold = {threshold}):")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  ROC-AUC:   {roc_auc:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    return y_pred, acc, roc_auc, precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/features.csv")
    parser.add_argument("--vit_csv", default="outputs/vit_embeddings.csv")
    parser.add_argument("--rppg_csv", default="outputs/rppg_features.csv")
    parser.add_argument("--model_out", default="outputs/checkpoints/model.joblib")
    parser.add_argument("--pca_components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--vit_mode", choices=["pca", "direct"], default="pca",
                         help="'pca': reduce 384-dim ViT embedding via fold-safe PCA "
                              "before the probe (original approach). 'direct': skip "
                              "PCA, fit the probe directly on the scaled 384-dim "
                              "embedding with stronger regularization.")
    parser.add_argument("--vit_c", type=float, default=None,
                         help="Override the ViT probe's LogisticRegression C "
                              "(inverse regularization strength). Defaults: "
                              f"{VIT_C_BY_MODE['pca']} for pca mode, "
                              f"{VIT_C_BY_MODE['direct']} for direct mode.")
    parser.add_argument("--calibration_bins", type=int, default=DEFAULT_CALIBRATION_BINS)
    parser.add_argument("--calibration_plot", default="outputs/calibration_curve_raw.png")
    parser.add_argument("--calibration_plot_platt", default="outputs/calibration_curve_platt.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(df["label"].value_counts())

    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise SystemExit(
            f"features.csv is missing expected columns: {missing_cols}. "
            f"Rerun media.py to regenerate features.csv with the current feature set."
        )

    use_rppg = False
    rppg_cols_present = []
    if os.path.exists(args.rppg_csv):
        rppg_df = pd.read_csv(args.rppg_csv)
        available = [c for c in RPPG_COLS if c in rppg_df.columns]
        if not available:
            print(f"{args.rppg_csv} found but has none of the expected columns {RPPG_COLS}")
        else:
            before = len(df)
            df = df.merge(rppg_df[["filename"] + available], on="filename", how="inner")
            print(f"Merged rPPG features from {args.rppg_csv}: {len(df)}/{before} videos matched")
            if len(df) < before:
                print(f"  Note: {before - len(df)} videos didn't have rPPG features and were dropped.")
            if len(df) > 0:
                use_rppg = True
                rppg_cols_present = available
    else:
        print(f"No rPPG features found at {args.rppg_csv} — training without them")

    use_vit = False
    vit_cols = []
    if os.path.exists(args.vit_csv):
        vit_df = pd.read_csv(args.vit_csv)
        vit_cols = [c for c in vit_df.columns if c.startswith("vit_")]
        before = len(df)
        df = df.merge(vit_df[["filename"] + vit_cols], on="filename", how="inner")
        print(f"Merged ViT embeddings ({len(vit_cols)}-dim) from {args.vit_csv}: "
              f"{len(df)}/{before} videos matched")
        if len(df) < before:
            print(f"  Note: {before - len(df)} videos dropped (missing motion or ViT features).")
        use_vit = len(df) > 0
        if use_vit:
            effective_c = args.vit_c if args.vit_c is not None else VIT_C_BY_MODE[args.vit_mode]
            if args.vit_mode == "pca":
                print(f"  ViT probe mode: PCA ({len(vit_cols)}-dim -> up to "
                      f"{args.pca_components} components), C={effective_c}")
            else:
                print(f"  ViT probe mode: DIRECT (full {len(vit_cols)}-dim embedding), "
                      f"C={effective_c}")
    else:
        print(f"No ViT embeddings found at {args.vit_csv} — training on motion features only")

    if len(df) == 0:
        raise SystemExit("No rows left after merging optional feature sets.")

    base_feature_cols = FEATURE_COLS + rppg_cols_present
    X = df[base_feature_cols].fillna(0).values
    y = (df["label"] == "fake").astype(int).values
    filenames = df["filename"].values
    VIT = df[vit_cols].values if use_vit else None
    active_feature_cols = base_feature_cols + ([VIT_PROBE_COL] if use_vit else [])

    loo = LeaveOneOut()
    y_true, y_proba_raw, y_proba_platt, held_out_names = [], [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if use_vit:
            vit_train_score = fold_safe_train_scores(
                VIT[train_idx], y_train, args.pca_components, args.vit_mode, args.vit_c
            )
            scaler, pca, probe = fit_vit_pipeline(
                VIT[train_idx], y_train, args.pca_components, args.vit_mode, args.vit_c
            )
            vit_test_score = vit_pipeline_scores(scaler, pca, probe, VIT[test_idx])
            X_train = np.hstack([X_train, vit_train_score.reshape(-1, 1)])
            X_test = np.hstack([X_test, vit_test_score.reshape(-1, 1)])

        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X_train, y_train)

        proba_raw_test = float(model.predict_proba(X_test)[0, 1])

        # Fold-safe Platt scaling on this outer fold's training rows
        n_train = len(y_train)
        k_inner = min(INNER_KFOLD_SPLITS, n_train)
        inner_kf = KFold(n_splits=max(2, k_inner), shuffle=True, random_state=42)
        raw_train_oof = np.zeros(n_train)
        for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
            inner_model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                eval_metric="logloss", random_state=42,
            )
            inner_model.fit(X_train[inner_train_idx], y_train[inner_train_idx])
            raw_train_oof[inner_val_idx] = inner_model.predict_proba(
                X_train[inner_val_idx]
            )[:, 1]

        platt = fit_platt_scaler(raw_train_oof, y_train)
        proba_platt_test = float(platt_transform(platt, np.array([proba_raw_test]))[0])

        y_true.append(y_test[0])
        y_proba_raw.append(proba_raw_test)
        y_proba_platt.append(proba_platt_test)
        held_out_names.append(filenames[test_idx[0]])

    y_true_arr = np.array(y_true)
    y_proba_raw_arr = np.array(y_proba_raw)
    y_proba_platt_arr = np.array(y_proba_platt)

    feature_desc = "+".join(
        filter(None, ["motion", "rPPG" if use_rppg else "",
                       f"ViT({args.vit_mode})" if use_vit else ""])
    )
    print(f"\n--- LOOCV results ({len(X)} videos, each held out once, {feature_desc}) ---")

    print("\n=== RAW (uncalibrated) ===")
    y_pred_raw, *_ = print_eval_metrics(y_true_arr, y_proba_raw_arr, DECISION_THRESHOLD, "raw")
    print("\nConfusion matrix (raw, rows=true, cols=pred) [real, fake]:")
    print(confusion_matrix(y_true_arr, y_pred_raw))
    print_calibration_curve(y_true_arr, y_proba_raw_arr, args.calibration_bins,
                             args.calibration_plot if args.calibration_plot else None, "raw")

    print("\n=== PLATT-CALIBRATED ===")
    y_pred_platt, *_ = print_eval_metrics(y_true_arr, y_proba_platt_arr, DECISION_THRESHOLD, "platt")
    print("\nConfusion matrix (platt, rows=true, cols=pred) [real, fake]:")
    print(confusion_matrix(y_true_arr, y_pred_platt))
    print_calibration_curve(y_true_arr, y_proba_platt_arr, args.calibration_bins,
                             args.calibration_plot_platt if args.calibration_plot_platt else None, "platt")

    # --- Final artifacts trained on ALL data ---
    final_scaler, final_pca, final_probe = None, None, None
    X_final = X
    if use_vit:
        final_train_scores = fold_safe_train_scores(
            VIT, y, args.pca_components, args.vit_mode, args.vit_c
        )
        final_scaler, final_pca, final_probe = fit_vit_pipeline(
            VIT, y, args.pca_components, args.vit_mode, args.vit_c
        )
        X_final = np.hstack([X, final_train_scores.reshape(-1, 1)])

    final_model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", random_state=42,
    )
    final_model.fit(X_final, y)

    n_all = len(y)
    k_inner_all = min(INNER_KFOLD_SPLITS, n_all)
    inner_kf_all = KFold(n_splits=max(2, k_inner_all), shuffle=True, random_state=42)
    raw_all_oof = np.zeros(n_all)
    for inner_train_idx, inner_val_idx in inner_kf_all.split(X_final):
        inner_model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
        )
        inner_model.fit(X_final[inner_train_idx], y[inner_train_idx])
        raw_all_oof[inner_val_idx] = inner_model.predict_proba(X_final[inner_val_idx])[:, 1]
    final_platt = fit_platt_scaler(raw_all_oof, y)

    importances = pd.Series(final_model.feature_importances_, index=active_feature_cols)
    print("\nFeature importance (from final model trained on all data):")
    print(importances.sort_values(ascending=False))

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    artifact = {
        "xgb_model": final_model,
        "feature_cols": base_feature_cols,
        "use_rppg": use_rppg,
        "rppg_cols": rppg_cols_present,
        "use_vit": use_vit,
        "vit_mode": args.vit_mode,
        "vit_c": args.vit_c if args.vit_c is not None else VIT_C_BY_MODE[args.vit_mode],
        "vit_scaler": final_scaler,
        "vit_pca": final_pca,  # None if vit_mode == 'direct'
        "vit_probe": final_probe,
        "vit_cols": vit_cols,
        "pca_components": args.pca_components,
        "platt_scaler": final_platt,
        "decision_threshold": DECISION_THRESHOLD,
    }
    joblib.dump(artifact, args.model_out)
    print(f"\nFinal model saved to {args.model_out}")
    print(f"ViT probe mode used: {args.vit_mode}")
    print("Note: predict.py should compute xgb_model.predict_proba(...)[:, 1], then")
    print("platt_scaler.predict_proba(raw_score.reshape(-1,1))[:, 1] to get the")
    print("calibrated probability. For the ViT probe: scaler.transform -> "
          "(pca.transform if vit_pca is not None else identity) -> probe.predict_proba.")


if __name__ == "__main__":
    main()