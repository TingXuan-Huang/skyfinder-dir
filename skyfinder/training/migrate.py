"""One-shot migration: flat results layout -> nested per-experiment subfolders.

Before May 2026 the trainer wrote `results/<run>.pt` and `results/<run>.json` at
the flat root. After May 2026 everything goes into `results/<subdir>/<run>.{pt,json}`
where `<subdir>` is the run name minus `_fold{N}` (and `_ep{N}`).

This module migrates any leftover flat-layout files into the nested layout.
It is idempotent.

Usage:
    python -m skyfinder.training.migrate --dry-run --root results/
    python -m skyfinder.training.migrate --root results/
    python -m skyfinder.training.migrate --repair-misrouted --dry-run
    python -m skyfinder.training.migrate --repair-misrouted
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .checkpoint import subdir_for


def _is_flat_artifact(path: Path) -> bool:
    """True if `path` is a flat-layout artifact (file at root of results_dir)."""
    if not path.is_file():
        return False
    return path.suffix in {".pt", ".json"}


def plan_migration(results_dir: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Return (planned_moves, in_flight_blockers).

    `planned_moves`: list of (src, dst) pairs to move (flat -> nested).
    `in_flight_blockers`: `_last.pt` files at the flat root, indicating a
        currently-running job. Migration aborts if any are found.
    """
    moves: list[tuple[Path, Path]] = []
    blockers: list[Path] = []
    if not results_dir.exists():
        return moves, blockers

    for child in sorted(results_dir.iterdir()):
        if not _is_flat_artifact(child):
            continue  # already nested or non-artifact
        stem = child.stem
        if stem.endswith("_last"):
            blockers.append(child)
            continue
        sub = subdir_for(stem)
        if sub == stem and "_fold" not in stem:
            # No `_fold{N}` suffix — leave file at root; new layout puts it in
            # `<stem>/<stem>.<suffix>` per `subdir_for`. Move anyway so the rule
            # is consistent.
            pass
        dst = results_dir / sub / child.name
        if dst == child or dst.exists():
            continue
        moves.append((child, dst))
    return moves, blockers


def run_migration(results_dir: Path, dry_run: bool = True) -> int:
    """Print and execute the migration. Returns exit code (0=ok, 2=blocked, 3=nothing-to-do)."""
    moves, blockers = plan_migration(results_dir)

    if blockers:
        print("[abort] In-flight job detected — refusing to migrate.", file=sys.stderr)
        for b in blockers:
            print(f"  {b}", file=sys.stderr)
        print("Wait for jobs to finish (or remove `_last.pt`) then re-run.", file=sys.stderr)
        return 2

    if not moves:
        print(f"[ok] {results_dir} is already nested — nothing to migrate.")
        return 3

    print(f"[plan] {len(moves)} file(s) to move:")
    for src, dst in moves:
        print(f"  {src.name}  ->  {dst.parent.name}/{dst.name}")

    if dry_run:
        print(f"[dry-run] No changes made. Re-run without --dry-run to execute.")
        return 0

    n_done = 0
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        n_done += 1
    print(f"[done] Moved {n_done} file(s).")
    return 0


def plan_misrouted_repair(
    results_dir: Path,
    random_results_dir: Path,
    cam_cond_results_dir: Path,
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Plan relocation of known nested directories written to the old root.

    Before explicit result-directory threading, random-control runs landed in
    ``results/<name>_rand`` and camera-conditioned runs landed in
    ``results/cam_cond_*``. These are already nested, so flat-to-nested
    migration cannot repair them. Destination conflicts are returned instead
    of being overwritten.
    """
    moves: list[tuple[Path, Path]] = []
    conflicts: list[Path] = []
    if not results_dir.exists():
        return moves, conflicts

    for source in sorted(results_dir.iterdir()):
        if not source.is_dir():
            continue
        if source.name.endswith("_rand"):
            destination = random_results_dir / source.name
        elif source.name.startswith("cam_cond_"):
            destination = cam_cond_results_dir / source.name
        else:
            continue
        if destination.exists():
            conflicts.append(destination)
        else:
            moves.append((source, destination))
    return moves, conflicts


def run_misrouted_repair(
    results_dir: Path,
    random_results_dir: Path,
    cam_cond_results_dir: Path,
    *,
    dry_run: bool = True,
) -> int:
    """Relocate known old-root random and camera-conditioned result folders."""
    moves, conflicts = plan_misrouted_repair(
        results_dir,
        random_results_dir,
        cam_cond_results_dir,
    )
    if conflicts:
        print("[abort] Destination already exists; refusing to overwrite:", file=sys.stderr)
        for destination in conflicts:
            print(f"  {destination}", file=sys.stderr)
        return 2
    if not moves:
        print("[ok] no known misrouted result directories found.")
        return 3

    print(f"[plan] relocate {len(moves)} misrouted result directory(s):")
    for source, destination in moves:
        print(f"  {source}  ->  {destination}")
    if dry_run:
        print("[dry-run] No changes made. Re-run without --dry-run to execute.")
        return 0

    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    print(f"[done] relocated {len(moves)} result directory(s).")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate flat results artifacts into nested run folders.")
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repair-misrouted",
        action="store_true",
        help="move old-root *_rand and cam_cond_* directories to their configured roots",
    )
    parser.add_argument("--random-root", type=Path, default=Path("results_random"))
    parser.add_argument("--cam-cond-root", type=Path, default=Path("results_cam_cond"))
    args = parser.parse_args()
    if args.repair_misrouted:
        raise SystemExit(
            run_misrouted_repair(
                args.root,
                args.random_root,
                args.cam_cond_root,
                dry_run=args.dry_run,
            )
        )
    raise SystemExit(run_migration(args.root, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
