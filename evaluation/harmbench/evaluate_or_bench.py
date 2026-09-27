#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.or_bench.evaluation import (
    OR_BENCH_RESPONSE_CHECKING_PROMPT,
    compute_category_metrics,
    extract_answer_category,
)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


API_MAX_RETRY = env_int("OR_BENCH_API_MAX_RETRY", 60)
API_RETRY_SLEEP = env_float("OR_BENCH_API_RETRY_SLEEP", 30.0)
API_TIMEOUT = env_float("OR_BENCH_API_TIMEOUT", 240.0)
API_MAX_SLEEP = env_float("OR_BENCH_API_MAX_SLEEP", 600.0)
API_RETRY_JITTER = env_float("OR_BENCH_API_RETRY_JITTER", 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OR-Bench completions with the benchmark-faithful response checker."
    )
    parser.add_argument("--behaviors_path", type=str, required=True)
    parser.add_argument("--completions_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--subset_name", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=["safe", "toxic"])
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api_base_url", type=str, default="")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--incremental_update", action="store_true")
    return parser.parse_args()


def load_behaviors(path: str) -> Dict[str, dict]:
    behaviors: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            behavior_id = row.get("BehaviorID") or row.get("Index")
            prompt = row.get("Behavior") or row.get("Goal")
            if behavior_id and prompt:
                behaviors[str(behavior_id)] = {
                    "prompt": prompt,
                    "category": row.get("Category", "unknown"),
                    "prompt_type": row.get("PromptType", ""),
                    "subset": row.get("Subset", ""),
                }
    if not behaviors:
        raise ValueError(f"No behaviors could be loaded from {path}")
    return behaviors


def load_completions(path: str) -> Dict[str, List[dict]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_existing_details(path: str) -> Dict[str, List[dict]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("details", {})


def build_client(api_key_env: str, api_base_url: str) -> OpenAI:
    dotenv_path = REPO_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} not found in environment variables or .env file.")
    kwargs = {"api_key": api_key}
    if api_base_url:
        kwargs["base_url"] = api_base_url
    return OpenAI(**kwargs)


def extract_retry_after_seconds(exc: Exception) -> float | None:
    message = str(exc)
    match = re.search(r"Please try again in ([0-9]+(?:\\.[0-9]+)?)s", message)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def judge_response(client: OpenAI, judge_model: str, prompt: str, response: str) -> Tuple[str, str]:
    for attempt in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {
                        "role": "user",
                        "content": OR_BENCH_RESPONSE_CHECKING_PROMPT.format(prompt, response),
                    }
                ],
                temperature=0.0,
                max_completion_tokens=128,
                timeout=API_TIMEOUT,
            )
            raw_output = completion.choices[0].message.content or ""
            return extract_answer_category(raw_output), raw_output
        except Exception as exc:
            print(f"API error ({attempt + 1}/{API_MAX_RETRY}): {exc}")
            if attempt == API_MAX_RETRY - 1:
                raise
            retry_after = extract_retry_after_seconds(exc)
            backoff_seconds = min(API_RETRY_SLEEP * (2 ** attempt), API_MAX_SLEEP)
            sleep_seconds = backoff_seconds
            if retry_after is not None:
                sleep_seconds = max(sleep_seconds, retry_after)
            if API_RETRY_JITTER > 0:
                sleep_seconds += random.uniform(0.0, API_RETRY_JITTER)
            time.sleep(sleep_seconds)
    raise RuntimeError("Unreachable retry state")


def compute_summary(details: Dict[str, List[dict]]) -> dict:
    answer_categories = []
    for entries in details.values():
        for entry in entries:
            answer_categories.append(entry.get("answer_category", "unknown"))
    return compute_category_metrics(answer_categories)


def main() -> None:
    args = parse_args()
    behaviors = load_behaviors(args.behaviors_path)
    completions = load_completions(args.completions_path)
    existing_details = load_existing_details(args.save_path) if args.incremental_update else {}
    client = build_client(args.api_key_env, args.api_base_url)

    details = dict(existing_details)
    pending_behavior_ids = []
    for behavior_id, completion_entries in completions.items():
        existing_entries = details.get(behavior_id, [])
        if args.incremental_update and len(existing_entries) == len(completion_entries) and existing_entries:
            continue
        pending_behavior_ids.append(behavior_id)

    print(
        f"Loaded {len(behaviors)} behaviors and {len(completions)} completion groups; "
        f"{len(pending_behavior_ids)} behavior IDs need OR-Bench judging."
    )

    for behavior_id in tqdm(pending_behavior_ids, desc="OR-Bench response checking"):
        behavior = behaviors.get(str(behavior_id))
        if behavior is None:
            print(f"Skipping unknown behavior ID: {behavior_id}")
            continue

        judged_entries = []
        for completion_entry in completions[behavior_id]:
            generation = completion_entry.get("generation", "")
            answer_category, raw_output = judge_response(
                client=client,
                judge_model=args.judge_model,
                prompt=behavior["prompt"],
                response=generation,
            )
            judged_entries.append(
                {
                    "original_prompt": behavior["prompt"],
                    "category": behavior["category"],
                    "generation": generation,
                    "answer_category": answer_category,
                    "answer_type_response": raw_output,
                }
            )
        details[str(behavior_id)] = judged_entries

        partial_summary = compute_summary(details)
        payload = {
            "benchmark": "OR-Bench",
            "subset_name": args.subset_name,
            "prompt_type": args.prompt_type,
            "judge_model": args.judge_model,
            "behaviors_path": args.behaviors_path,
            "completions_path": args.completions_path,
            **partial_summary,
            "details": details,
        }
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
        with open(args.save_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    final_summary = compute_summary(details)
    final_payload = {
        "benchmark": "OR-Bench",
        "subset_name": args.subset_name,
        "prompt_type": args.prompt_type,
        "judge_model": args.judge_model,
        "behaviors_path": args.behaviors_path,
        "completions_path": args.completions_path,
        **final_summary,
        "details": details,
    }
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
    with open(args.save_path, "w", encoding="utf-8") as handle:
        json.dump(final_payload, handle, indent=2)

    print(
        f"Saved OR-Bench results to {args.save_path} "
        f"(subset={args.subset_name}, rejection_rate={final_summary['rejection_rate']:.4f}, n={final_summary['n']})."
    )


if __name__ == "__main__":
    main()
