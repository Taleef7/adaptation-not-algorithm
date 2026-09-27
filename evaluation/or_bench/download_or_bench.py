#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.or_bench.common import OR_BENCH_DATA_DIR, OR_BENCH_SUBSETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize OR-Bench subsets.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing normalized CSV files if they already exist.",
    )
    return parser.parse_args()


def build_behavior_rows(rows: Iterable[dict], subset_name: str, prompt_type: str) -> List[dict]:
    normalized_rows = []
    slug = subset_name.lower().replace("-", "_")
    for index, row in enumerate(rows, start=1):
        normalized_rows.append(
            {
                "BehaviorID": f"or_bench_{slug}_{index:04d}",
                "Behavior": row["prompt"],
                "Category": row["category"],
                "SourceBenchmark": "OR-Bench",
                "Subset": subset_name,
                "PromptType": prompt_type,
            }
        )
    return normalized_rows


def main() -> None:
    args = parse_args()
    OR_BENCH_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for subset_name, spec in OR_BENCH_SUBSETS.items():
        output_path = OR_BENCH_DATA_DIR / spec["csv_name"]
        if output_path.exists() and not args.force:
            print(f"Skipping existing file: {output_path}")
            continue

        dataset = load_dataset("bench-llms/or-bench", spec["hf_config"], split="train")
        rows = build_behavior_rows(dataset, subset_name=subset_name, prompt_type=spec["prompt_type"])
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")
        print(df["Category"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
