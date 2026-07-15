"""
train.py
Loads outputs/features.csv, trains XGBoost classifier, reports accuracy.

Usage:
    python src/train.py --csv outputs/features.csv
"""

import argparse
import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os


FEATURE_COLS = [
    "blink_rate", "ear_mean", "ear_std",
    "jaw_velocity_mean", "jaw_velocity_std", "jaw_jitter_fft_energy",
    "mouth_velocity_mean", "mouth_velocity_std", "mouth_jitter_fft_energy",
    "overall_velocity_mean", "overall_velocity_std", "overall_jitter_fft_energy",
    "av_sync_lag_ms", "av_sync_confidence",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/features.csv")
    parser.add_argument("--model_out", default="outputs/checkpoints/model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")
    print(df["label"].value_counts())

    X = df[FEATURE_COLS].fillna(0).values
    y = (df["label"] == "fake").astype(int).values  # 1 = fake, 0 = real
    filenames = df["filename"].values

    # Leave-One-Out CV: train on all-but-one video, test on the held-out one,
    # repeat for every video. With ~13 samples this is the only honest way
    # to use every row for both training and testing without leakage.
    loo = LeaveOneOut()
    y_true, y_pred, held_out_names = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        y_true.append(y_test[0])
        y_pred.append(pred)
        held_out_names.append(filenames[test_idx[0]])

    acc = accuracy_score(y_true, y_pred)
    print(f"\nLOOCV accuracy ({len(X)} videos, each held out once): {acc:.3f} "
          f"({sum(np.array(y_true) == np.array(y_pred))}/{len(y_true)} correct)")

    print("\nPer-video result:")
    for name, t, p in zip(held_out_names, y_true, y_pred):
        true_label = "fake" if t == 1 else "real"
        pred_label = "fake" if p == 1 else "real"
        mark = "correct" if t == p else "WRONG"
        print(f"  {name:40s} true={true_label:5s} pred={pred_label:5s} [{mark}]")

    print("\nConfusion matrix (rows=true, cols=pred) [real, fake]:")
    print(confusion_matrix(y_true, y_pred))

    # Train final model on ALL data for actual use in predict.py
    # (LOOCV above only measures accuracy — this final fit is what gets saved)
    final_model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", random_state=42,
    )
    final_model.fit(X, y)

    importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLS)
    print("\nFeature importance (from final model trained on all data):")
    print(importances.sort_values(ascending=False))

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(final_model, args.model_out)
    print(f"\nFinal model (trained on all {len(X)} videos) saved to {args.model_out}")
    print("Note: LOOCV accuracy above is your honest performance estimate — "
          "the saved model itself was fit on everything, so don't re-test it "
          "on training videos and call that accuracy.")


if __name__ == "__main__":
    main()
