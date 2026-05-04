"""
Sample several FEVER claims and compare query generation outputs from:
1. BM25 + Claim
2. BM25 + Entity
3. LLM Prompt + BM25 (Same Prompt, No GRPO)

Example usage:

python inspect_query_examples.py \
  --model-path /home/jovyan/models/Qwen2.5-1.5B-Instruct \
  --num-examples 5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from config import PathConfig
from query_parser import claim_only_query, entity_query
from eval_baseline import (
    load_eval_data,
    make_llm_prompt_query_fn,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Default: valid_formal_subset.parquet if exists.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Base model path for LLM Prompt baseline.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/query_examples.csv",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    path_cfg = PathConfig()

    # default evaluation file
    if args.input_path:
        input_path = Path(args.input_path)
    else:
        formal_subset = path_cfg.export_dir / "valid_formal_subset.parquet"

        if formal_subset.exists():
            input_path = formal_subset
        elif path_cfg.valid_for_verl_path.exists():
            input_path = path_cfg.valid_for_verl_path
        else:
            input_path = path_cfg.dev_path

    print(f"[INFO] Loading data from: {input_path}")

    data = load_eval_data(input_path, max_examples=None)

    if len(data) == 0:
        raise ValueError("Dataset is empty.")

    sampled = random.sample(
        data,
        min(args.num_examples, len(data))
    )

    print("[INFO] Loading LLM baseline model...")

    llm_query_fn = make_llm_prompt_query_fn(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        max_new_tokens=32,
    )

    rows = []

    for idx, ex in enumerate(sampled, start=1):

        claim = ex["claim"]

        q_claim = claim_only_query(ex)

        q_entity = entity_query(ex)

        q_llm = llm_query_fn(ex)

        row = {
            "example_id": idx,
            "claim": claim,
            "bm25_claim_query": q_claim,
            "bm25_entity_query": q_entity,
            "llm_prompt_query": q_llm,
        }

        rows.append(row)

        print("\n" + "=" * 80)
        print(f"[Example {idx}]")
        print(f"Claim:\n{claim}\n")

        print("[BM25 + Claim]")
        print(q_claim)

        print("\n[BM25 + Entity]")
        print(q_entity)

        print("\n[LLM Prompt + BM25]")
        print(q_llm)

    df = pd.DataFrame(rows)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print(f"[DONE] Saved results to: {output_path}")


if __name__ == "__main__":
    main()