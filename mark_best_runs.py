#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import pandas as pd
import re

MODELS = ("roberta-base", "roberta-large")
VALID_PREFIXES = ("full_finetuning", "lora", "rslora")

def parse_run(folder_name: str):
    """
    Returns (run_type, rank_or_none).
    run_type in {"full_finetuning", "lora", "rslora"}.
    rank_or_none is int for lora/rslora, None for full_finetuning.
    """
    if folder_name.startswith("full_finetuning_"):
        return ("full_finetuning", None)

    m = re.match(r"(lora|rslora)_r(\d+)_", folder_name)
    if m:
        return (m.group(1), int(m.group(2)))

    return (None, None)

def get_epoch5_accuracy(csv_path: Path):
    """
    Read eval_per_epoch.csv and return accuracy at epoch==5 (float).
    If no exact epoch==5, try the 5th row (index 4). If <5 rows, return None.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if "accuracy" not in df.columns:
        return None

    # Prefer explicit epoch==5 (handles 5 or 5.0)
    if "epoch" in df.columns:
        match = df[df["epoch"].astype(float) == 5.0]
        if len(match) == 1:
            return float(match["accuracy"].iloc[0])
        # If multiple or none, fall through to row-based check

    if len(df) >= 5:
        return float(df["accuracy"].iloc[4])

    return None

def safe_rename(src: Path, suffix="_best_accuracy"):
    """Rename src directory by appending suffix; avoid duplicates/collisions."""
    if src.name.endswith(suffix):
        return src  # already tagged

    candidate = src.parent / f"{src.name}{suffix}"
    if not candidate.exists():
        src.rename(candidate)
        return candidate

    i = 1
    while True:
        c = src.parent / f"{src.name}{suffix}-{i}"
        if not c.exists():
            src.rename(c)
            return c
        i += 1

def main():
    ap = argparse.ArgumentParser(description="Mark best-accuracy runs by renaming folders.")
    ap.add_argument("task", help="Task name (e.g., rte, cola)")
    ap.add_argument("--root", default=".", help="Root path containing the task dir (default: .)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be renamed without changing anything")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    task_dir = root / args.task
    if not task_dir.is_dir():
        print(f"❗ Task dir not found: {task_dir}")
        return

    print(f"=== Scan: {task_dir} ===")

    issues = []
    decisions = []

    for model in MODELS:
        model_dir = task_dir / model
        if not model_dir.is_dir():
            issues.append(f"Missing model dir: {model_dir}")
            continue

        # Collect runs into groups: (run_type, rank) -> list[(path, acc5)]
        groups = {}
        for run_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            run_type, rank = parse_run(run_dir.name)
            if run_type is None:
                continue

            csv_path = run_dir / "eval_per_epoch.csv"
            if not csv_path.exists():
                issues.append(f"[{model}] {run_dir.name}: eval_per_epoch.csv not found")
                continue

            acc5 = get_epoch5_accuracy(csv_path)
            if acc5 is None:
                issues.append(f"[{model}] {run_dir.name}: no usable 5th-epoch accuracy")
                continue

            groups.setdefault((run_type, rank), []).append((run_dir, acc5))

        # Decide best per group
        for (run_type, rank), entries in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1] if x[0][1] is not None else -1)):
            # Pick max-accuracy run
            best_dir, best_acc = max(entries, key=lambda t: t[1])
            label = f"{model} | {run_type}" + (f"_r{rank}" if rank is not None else "")
            decisions.append((label, best_dir, best_acc))

    # Print decisions and apply
    if decisions:
        print("\n--- Best by 5th-epoch accuracy ---")
        for label, best_dir, best_acc in decisions:
            print(f"{label:<28} -> {best_dir.name}  (acc={best_acc:.6f})")
        if args.dry_run:
            print("\n(DRY RUN) No folders were renamed.")
        else:
            print("\nRenaming best runs (appending _best_accuracy)…")
            for _, best_dir, _ in decisions:
                try:
                    new_path = safe_rename(best_dir)
                    print(f"RENAMED: {best_dir.name} -> {new_path.name}")
                except Exception as e:
                    issues.append(f"Rename failed for {best_dir}: {e}")
    else:
        print("\nNo eligible runs found.")

    # Issues at the end
    if issues:
        print("\n--- Issues ---")
        for s in issues:
            print(s)
    else:
        print("\nAll good ✅")

if __name__ == "__main__":
    main()
