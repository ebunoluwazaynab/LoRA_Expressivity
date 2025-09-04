#!/usr/bin/env python3
import argparse
from pathlib import Path

# Expected data-row counts (exclude header)
EXPECT = {
    "roberta-base": {
        "full_finetuning": 770,
        "lora": 1115,
        "rslora": 1115,
    },
    "roberta-large": {
        "full_finetuning": 1490,
        "lora": 2195,
        "rslora": 2195,
    },
}

VALID_PREFIXES = ("full_finetuning", "lora", "rslora")
MODELS = ("roberta-base", "roberta-large")


def count_data_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        # subtract 1 for header
        return sum(1 for _ in f) - 1


def classify_run(run_name: str) -> str | None:
    for p in VALID_PREFIXES:
        if run_name.startswith(p + "_"):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Check sr_layerwise.csv row counts.")
    ap.add_argument("task", help="Task folder name, e.g. rte or cola")
    ap.add_argument("--root", default=".", help="Root directory (default: current dir)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    task_dir = root / args.task
    if not task_dir.is_dir():
        print(f"❗ Task dir not found: {task_dir}")
        return

    print(f"=== Checking: {task_dir} ===\n")

    issues = []  # collect not found / mismatched cases

    for model in MODELS:
        model_dir = task_dir / model
        if not model_dir.is_dir():
            issues.append(f"MISMATCH | model dir: {model_dir}")
            continue

        print(f">> {model}")
        for run_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            run_type = classify_run(run_dir.name)
            if run_type is None:
                continue  # ignore unrelated folders

            expected = EXPECT[model][run_type]
            csv_path = run_dir / "sr_layerwise.csv"

            if not csv_path.exists():
                issues.append(f"MISMATCH | {run_dir.relative_to(root)} -> sr_layerwise.csv")
                continue

            rows = count_data_rows(csv_path)
            if rows == expected:
                print(f"MATCH     | {run_dir.name:<35} rows={rows} (expected={expected})")
            else:
                issues.append(
                    f"MISMATCH  | {run_dir.relative_to(root)} rows={rows} (expected={expected})"
                )
        print()

    print("=== Summary ===")
    if issues:
        print("\n--- Issues ---")
        for i in issues:
            print(i)
    else:
        print("All runs OK ✅")


if __name__ == "__main__":
    main()
