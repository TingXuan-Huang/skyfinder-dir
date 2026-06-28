"""C2-metadata x CNN ensemble: does the image add anything over metadata?

prediction = alpha*C2 + (1-alpha)*CNN on the TEST split; sweep alpha and find the
MAE-minimising blend. alpha~1 -> metadata suffices; alpha~0 -> image suffices;
alpha~0.5 -> complementary (image+metadata fusion is the right project).

C2 is refit per fold here to get raw metadata-only predictions; CNN predictions
are read from each run's JSON. ``--selection validation`` chooses alpha on the
validation split and reports one untouched test result per fold.

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

from skyfinder.training.checkpoint import load_results, subdir_for
from skyfinder.training.engine import per_bin_mae
from skyfinder.training.splits import load_splits
from skyfinder.analysis.baselines_metadata import METADATA_FEATURES, make_metadata_regressor

ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 2)


def c2_pred(df: pd.DataFrame, fold: dict, split: str, seed: int = 0) -> np.ndarray:
    """Fit C2 on train and predict one configured fold split."""
    train_df, target_df = df.iloc[fold["train"]], df.iloc[fold[split]]
    model = make_metadata_regressor(seed=seed)
    model.fit(train_df[METADATA_FEATURES], train_df["TempM"].to_numpy())
    return model.predict(target_df[METADATA_FEATURES])


def c2_test_pred(df: pd.DataFrame, fold: dict, seed: int = 0) -> np.ndarray:
    """Backward-compatible C2 test prediction helper."""
    return c2_pred(df, fold, "test", seed)


def blend_sweep(c2, cnn, y, train_y, bin_w=1.0) -> dict:
    """alpha -> per-bin MAE of (alpha*c2 + (1-alpha)*cnn)."""
    return {a: per_bin_mae(y, a * c2 + (1 - a) * cnn, train_y, bin_w=bin_w) for a in ALPHAS}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", required=True, help="CNN run-name prefix, e.g. lds_fds_resnet50")
    ap.add_argument("--results", default="results")
    ap.add_argument("--labels", default="data/labels_with_images.csv")
    ap.add_argument("--splits", default="data/splits/loco_5fold.json")
    ap.add_argument("--bin-w", type=float, default=1.0)
    ap.add_argument("--selection", choices=["test_sweep", "validation"], default="test_sweep")
    ap.add_argument("--out", default=None, help="optional JSON output for the selected blend")
    args = ap.parse_args()

    df = _prep(pd.read_csv(args.labels))
    splits = load_splits(args.splits, args.labels, len(df))

    per_alpha = {a: [] for a in ALPHAS}
    selected = []
    for fold in splits:
        name = f"{args.cnn}_fold{fold['fold']}"
        jp = Path(args.results) / subdir_for(name) / f"{name}.json"
        if not jp.exists():
            continue
        d = load_results(jp)
        if not d.get("test_preds"):
            continue
        train_y = df["TempM"].to_numpy()[fold["train"]]
        cnn_test = np.asarray(d["test_preds"], float)
        y_test = np.asarray(d["test_ys"], float)
        c2_test = c2_pred(df, fold, "test")
        expected_test = df["TempM"].to_numpy()[fold["test"]]
        if cnn_test.shape != c2_test.shape or y_test.shape != c2_test.shape:
            raise ValueError(f"{name}: prediction lengths do not match the configured test fold")
        if not np.allclose(y_test, expected_test):
            raise ValueError(f"{name}: saved test targets do not match the configured labels/splits")
        sweep = blend_sweep(c2_test, cnn_test, y_test, train_y, args.bin_w)
        for a in ALPHAS:
            per_alpha[a].append(sweep[a]["overall"])

        if args.selection == "validation":
            if not d.get("val_preds"):
                raise ValueError(f"{name}: validation predictions are required for validation selection")
            cnn_val = np.asarray(d["val_preds"], float)
            y_val = np.asarray(d["val_ys"], float)
            c2_val = c2_pred(df, fold, "val")
            expected_val = df["TempM"].to_numpy()[fold["val"]]
            if cnn_val.shape != c2_val.shape or not np.allclose(y_val, expected_val):
                raise ValueError(f"{name}: saved validation predictions do not match the configured fold")
            val_sweep = blend_sweep(c2_val, cnn_val, y_val, train_y, args.bin_w)
            alpha = min(ALPHAS, key=lambda value: val_sweep[value]["overall"])
            test_metric = per_bin_mae(y_test, alpha * c2_test + (1 - alpha) * cnn_test, train_y, args.bin_w)
            selected.append({
                "fold": int(fold["fold"]),
                "alpha": float(alpha),
                "val": val_sweep[alpha],
                "test": test_metric,
            })

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
    if selected:
        test = np.asarray([row["test"]["overall"] for row in selected])
        report = {
            "method": "validation_selected_ensemble",
            "cnn": args.cnn,
            "per_fold": selected,
            "summary": {
                "test_mae_mean": float(test.mean()),
                "test_mae_std": float(test.std(ddof=1)),
                "alphas": [row["alpha"] for row in selected],
            },
        }
        print(f"[validation-selected] test={test.mean():.3f} +/- {test.std(ddof=1):.3f}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=2))
            print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
