"""C2-metadata x CNN ensemble: does the image add anything over metadata?

prediction = alpha*C2 + (1-alpha)*CNN on the TEST split; sweep alpha and find the
MAE-minimising blend. alpha~1 -> metadata suffices; alpha~0 -> image suffices;
alpha~0.5 -> complementary (image+metadata fusion is the right project).

C2 (HistGBR on CamId/Hour/Month/Lat/Long) is refit per fold here to get raw predictions;
CNN test predictions are read from each run's JSON (written by infer_test.py).

Usage:
    python -m skyfinder.analysis.ensemble --cnn lds_fds_resnet50 --results results \
        --labels data/labels_with_images.csv --splits data/splits/loco_5fold.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfinder.training.checkpoint import subdir_for
from skyfinder.training.engine import per_bin_mae

FEATURES = ["CamId_cat", "Hour", "Month", "Latitude", "Longitude"]
ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 2)


def c2_test_pred(df: pd.DataFrame, fold: dict, seed: int = 0) -> np.ndarray:
    """Refit the metadata GBM on the fold's train, predict on its test (raw preds)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    train_df, test_df = df.iloc[fold["train"]], df.iloc[fold["test"]]
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=6,
                                          categorical_features=[0], random_state=seed)
    model.fit(train_df[FEATURES].to_numpy(), train_df["TempM"].to_numpy())
    return model.predict(test_df[FEATURES].to_numpy())


def blend_sweep(c2, cnn, y, train_y, bin_w=1.0) -> dict:
    """alpha -> per-bin MAE of (alpha*c2 + (1-alpha)*cnn)."""
    return {a: per_bin_mae(y, a * c2 + (1 - a) * cnn, train_y, bin_w=bin_w) for a in ALPHAS}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    cam_dtype = pd.CategoricalDtype(categories=sorted(df["CamId"].unique()))
    df = df.copy()
    df["CamId_cat"] = df["CamId"].astype(cam_dtype).cat.codes
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", required=True, help="CNN run-name prefix, e.g. lds_fds_resnet50")
    ap.add_argument("--results", default="results")
    ap.add_argument("--labels", default="data/labels_with_images.csv")
    ap.add_argument("--splits", default="data/splits/loco_5fold.json")
    ap.add_argument("--bin-w", type=float, default=1.0)
    args = ap.parse_args()

    df = _prep(pd.read_csv(args.labels))
    splits = json.loads(Path(args.splits).read_text())

    per_alpha = {a: [] for a in ALPHAS}
    for fold in splits:
        name = f"{args.cnn}_fold{fold['fold']}"
        jp = Path(args.results) / subdir_for(name) / f"{name}.json"
        if not jp.exists():
            continue
        d = json.load(open(jp))
        if not d.get("test_preds"):
            continue
        cnn = np.asarray(d["test_preds"], float)
        y = np.asarray(d["test_ys"], float)
        c2 = c2_test_pred(df, fold)
        train_y = df["TempM"].to_numpy()[fold["train"]]
        sweep = blend_sweep(c2, cnn, y, train_y, args.bin_w)
        for a in ALPHAS:
            per_alpha[a].append(sweep[a]["overall"])

    if not per_alpha[ALPHAS[0]]:
        print(f"no CNN test preds for '{args.cnn}' — run infer_test.py first")
        return
    print(f"=== alpha sweep (test overall MAE, mean over folds), CNN={args.cnn} ===")
    print("alpha=0 -> pure CNN | alpha=1 -> pure C2 metadata")
    best_a, best = None, 1e9
    for a in ALPHAS:
        m = float(np.mean(per_alpha[a]))
        if m < best:
            best, best_a = m, a
        print(f"  alpha={a:.1f}: {m:.3f}")
    print(f"[best] alpha={best_a} -> {best:.3f}  "
          f"(CNN-only={np.mean(per_alpha[0.0]):.3f}, C2-only={np.mean(per_alpha[1.0]):.3f})")


if __name__ == "__main__":
    main()
