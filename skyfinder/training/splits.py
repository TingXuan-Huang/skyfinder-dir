"""Split-manifest creation and validation.

SkyFinder folds contain positional row indices, so a reordered label CSV silently
changes the train/validation/test membership. New split files record a SHA-256
fingerprint of the CSV they were built from; consumers validate it before using
the indices. Legacy list-only split files remain readable with a warning so
existing experiments can be migrated deliberately.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path


SPLIT_SCHEMA_VERSION = 1


def file_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Return a content hash without loading a potentially large CSV into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def make_split_manifest(folds: list[dict], labels_path: Path | str, n_rows: int) -> dict:
    """Wrap positional folds with the identity of the source labels file."""
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "labels_sha256": file_sha256(labels_path),
        "n_rows": int(n_rows),
        "folds": folds,
    }


def load_splits(splits_path: Path | str, labels_path: Path | str, n_rows: int) -> list[dict]:
    """Load and validate a split manifest against the labels CSV.

    Legacy list-only files are accepted for backward compatibility but emit a
    warning because their source row order cannot be verified. Re-run either
    ``data/splits.py`` or ``data/splits_random.py`` to upgrade them.
    """
    payload = json.loads(Path(splits_path).read_text())
    if isinstance(payload, list):
        warnings.warn(
            f"legacy split file without a labels fingerprint: {splits_path}; regenerate it before "
            "reporting final results",
            RuntimeWarning,
            stacklevel=2,
        )
        return payload

    if not isinstance(payload, dict) or payload.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(f"unsupported split manifest: {splits_path}")
    if payload.get("n_rows") != int(n_rows):
        raise ValueError(
            f"split row count ({payload.get('n_rows')}) does not match labels ({n_rows}); regenerate splits"
        )
    if payload.get("labels_sha256") != file_sha256(labels_path):
        raise ValueError(f"split labels fingerprint does not match {labels_path}; regenerate splits")
    folds = payload.get("folds")
    if not isinstance(folds, list):
        raise ValueError(f"split manifest has no fold list: {splits_path}")
    return folds


def get_fold(splits: list[dict], fold_id: int) -> dict:
    """Return the uniquely recorded fold instead of assuming list position equals ID."""
    matches = [fold for fold in splits if fold.get("fold") == fold_id]
    if len(matches) != 1:
        raise ValueError(f"expected one fold with id {fold_id}, found {len(matches)}")
    return matches[0]
