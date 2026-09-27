from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLKIT_ROOT = REPO_ROOT / "toolkits" / "HarmBench"
ASR_BY_ATTACK_DIR = REPO_ROOT / "ASRs by attack"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OR_BENCH_DATA_DIR = TOOLKIT_ROOT / "data" / "behavior_datasets" / "or_bench"
OR_BENCH_RESULTS_DIR = TOOLKIT_ROOT / "results" / "ORBench"
CANONICAL_MODEL_SOURCE = ASR_BY_ATTACK_DIR / "benign_orr_all_evaluators.csv"

MODEL_PREFIX_BY_FAMILY = {
    "gemma2": "gemma2",
    "llama31": "llama31",
    "phi4": "phi4",
    "qwen3": "qwen3",
    "qwen25": "qwen25_14b",
}

OR_BENCH_SUBSETS: Dict[str, Dict[str, str]] = {
    "Hard-1K": {
        "hf_config": "or-bench-hard-1k",
        "csv_name": "or_bench_hard_1k.csv",
        "prompt_type": "safe",
        "summary_prefix": "hard_1k",
    },
    "Toxic": {
        "hf_config": "or-bench-toxic",
        "csv_name": "or_bench_toxic.csv",
        "prompt_type": "toxic",
        "summary_prefix": "toxic",
    },
}


def model_name_from_family_config(model_family: str, config: str) -> str:
    prefix = MODEL_PREFIX_BY_FAMILY[model_family]
    return f"{prefix}_{config}_custom"


def canonical_model_records() -> List[dict]:
    if CANONICAL_MODEL_SOURCE.exists():
        df = (
            pd.read_csv(CANONICAL_MODEL_SOURCE, usecols=["model_family", "config"])
            .drop_duplicates()
            .sort_values(["model_family", "config"])
        )
        return [
            {
                "model_family": row.model_family,
                "config": row.config,
                "model_name": model_name_from_family_config(row.model_family, row.config),
            }
            for row in df.itertuples(index=False)
        ]

    fallback = [
        ("gemma2", "base_fp16"),
        ("gemma2", "base_4bit"),
        ("gemma2", "lora"),
        ("gemma2", "qlora"),
        ("gemma2", "fft"),
        ("llama31", "base_fp16"),
        ("llama31", "base_4bit"),
        ("llama31", "lora"),
        ("llama31", "qlora"),
        ("llama31", "fft"),
        ("phi4", "base_fp16"),
        ("phi4", "base_4bit"),
        ("phi4", "lora"),
        ("phi4", "qlora"),
        ("phi4", "fft"),
        ("qwen25", "base_fp16"),
        ("qwen25", "base_4bit"),
        ("qwen25", "lora"),
        ("qwen25", "qlora"),
        ("qwen25", "fft"),
        ("qwen3", "base_fp16"),
        ("qwen3", "base_4bit"),
        ("qwen3", "lora"),
        ("qwen3", "qlora"),
        ("qwen3", "fft"),
    ]
    return [
        {
            "model_family": model_family,
            "config": config,
            "model_name": model_name_from_family_config(model_family, config),
        }
        for model_family, config in fallback
    ]


def subset_csv_path(subset_name: str) -> Path:
    spec = OR_BENCH_SUBSETS[subset_name]
    return OR_BENCH_DATA_DIR / spec["csv_name"]


def subset_results_dir(subset_name: str) -> Path:
    return OR_BENCH_RESULTS_DIR / subset_name / "DirectRequest" / "default"

