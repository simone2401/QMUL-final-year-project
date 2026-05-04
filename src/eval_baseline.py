"""Evaluate fixed and LLM-prompt baselines against the pre-built sharded BM25 corpus.

Baselines:
1. BM25 + Claim: use the raw claim as the query.
2. BM25 + Entity: use rule-based entity/query-term expansion.
3. LLM Prompt + BM25: use the same prompt as GRPO, but the base model is not GRPO-trained.

The script supports both FEVER jsonl and verl parquet inputs. By default, it prefers
outputs/valid_formal_subset.parquet when it exists, so baseline evaluation can match
the formal GRPO validation subset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

from bm25_env import ShardedBM25SentenceEnv, load_jsonl
from config import PathConfig, RewardWeights, REWARD_FORMULA, build_messages, ensure_dirs
from query_parser import claim_only_query, entity_query, parse_model_query_output, tokenize_for_bm25
from reward_fn import (
    compute_retrieval_metrics,
    compute_retrieval_metrics_from_gold,
    compute_reward,
    summarize_runs,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def require_corpus(path_cfg: PathConfig) -> ShardedBM25SentenceEnv:
    return ShardedBM25SentenceEnv.load(path_cfg)


def _strategy_name(strategy_fn: Callable[[dict], str]) -> str:
    names = {
        claim_only_query.__name__: "BM25 + Claim",
        entity_query.__name__: "BM25 + Entity",
        "llm_prompt_query": "LLM Prompt + BM25 (Same Prompt, No GRPO)",
    }
    return names.get(getattr(strategy_fn, "__name__", "strategy"), getattr(strategy_fn, "__name__", "strategy"))


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.size == 0
        if isinstance(value, np.generic):
            return False
    except Exception:
        pass
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _to_plain(value: Any) -> Any:
    """Convert parquet/numpy values into plain Python JSON-like values."""
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return [_to_plain(x) for x in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(v) for v in value]
    return value


def _maybe_json_load(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def _normalise_record(row: dict) -> dict:
    """Make parquet/json rows compatible with the original baseline code."""
    row = _to_plain(dict(row))
    for key in ["ground_truth", "extra_info", "reward_model", "meta", "prompt"]:
        if key in row:
            row[key] = _to_plain(_maybe_json_load(row[key]))
    return row

#Return the first non-empty value among candidates
def _first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        value = _to_plain(_maybe_json_load(value))
        if not _is_empty_value(value):
            return value
    return default


def _extract_ground_truth(ex: dict) -> tuple[list[str], list[str], list[list[str]]] | None:
    if "evidence" in ex:
        return None

    ground_truth = _to_plain(_maybe_json_load(ex.get("ground_truth", {})))
    extra_info = _to_plain(_maybe_json_load(ex.get("extra_info", {})))
    if not isinstance(ground_truth, dict):
        ground_truth = {}
    if not isinstance(extra_info, dict):
        extra_info = {}

    gold_doc_ids = _first_present(extra_info.get("gold_doc_ids"), ground_truth.get("gold_doc_ids"), default=[])
    gold_page_ids = _first_present(extra_info.get("gold_page_ids"), ground_truth.get("gold_page_ids"), default=[])
    gold_evidence_sets = _first_present(
        extra_info.get("gold_evidence_sets"),
        ground_truth.get("gold_evidence_sets"),
        default=[],
    )

    return list(_to_plain(gold_doc_ids)), list(_to_plain(gold_page_ids)), list(_to_plain(gold_evidence_sets))


def _compute_metrics(ex: dict, retrieved: list[str], local_page_map: dict[str, str]) -> dict[str, float]:
    extracted = _extract_ground_truth(ex)
    if extracted is None:
        return compute_retrieval_metrics(ex, retrieved, local_page_map)
    gold_doc_ids, gold_page_ids, gold_evidence_sets = extracted
    return compute_retrieval_metrics_from_gold(
        retrieved_doc_ids=retrieved,
        doc_page_map=local_page_map,
        gold_doc_ids=gold_doc_ids,
        gold_page_ids=gold_page_ids,
        gold_evidence_sets=gold_evidence_sets,
    )


def _get_claim(ex: dict) -> str:
    ground_truth = _to_plain(_maybe_json_load(ex.get("ground_truth", {})))
    extra_info = _to_plain(_maybe_json_load(ex.get("extra_info", {})))
    if not isinstance(ground_truth, dict):
        ground_truth = {}
    if not isinstance(extra_info, dict):
        extra_info = {}
    claim = _first_present(ex.get("claim"), extra_info.get("claim"), ground_truth.get("claim"), default="")
    return str(claim)


def evaluate_once(
    data: List[dict],
    env: ShardedBM25SentenceEnv,
    strategy_fn: Callable[[dict], str],
    reward_weights: RewardWeights | None = None,
    eval_topk: int = 10,
    *,
    show_progress: bool = True,
    progress_every: int = 10,
    collect_details: bool = False,
) -> Dict[str, float] | tuple[Dict[str, float], List[dict]]:
    totals = {
        "Sentence Recall@5": 0.0,
        "Sentence Recall@10": 0.0,
        "Page Recall@5": 0.0,
        "Page Recall@10": 0.0,
        "Full Evidence@10": 0.0,
        "Reward": 0.0,
        "MRR": 0.0,
    }
    details: List[dict] = []

    strategy_name = _strategy_name(strategy_fn)
    iterator = data
    use_tqdm = show_progress and tqdm is not None

    if use_tqdm:
        iterator = tqdm(data, desc=f"Evaluating {strategy_name}", unit="example")
    elif show_progress:
        print(f"[start] {strategy_name}: 0/{len(data)}")

    for idx, ex in enumerate(iterator, start=1):
        query = strategy_fn(ex)
        retrieved, local_page_map = env.retrieve_ids_with_page_map(query, topk=eval_topk)
        metrics = _compute_metrics(ex, retrieved, local_page_map)
        reward = compute_reward(metrics, weights=reward_weights)
        totals["Sentence Recall@5"] += metrics["sentence_recall@5"]
        totals["Sentence Recall@10"] += metrics["sentence_recall@10"]
        totals["Page Recall@5"] += metrics["page_recall@5"]
        totals["Page Recall@10"] += metrics["page_recall@10"]
        totals["Full Evidence@10"] += metrics["full_evidence@10"]
        totals["Reward"] += reward
        totals["MRR"] += metrics["mrr"]

        if collect_details:
            extra_info = ex.get("extra_info", {}) if isinstance(ex.get("extra_info", {}), dict) else {}
            details.append(
                {
                    "id": _to_plain(_first_present(ex.get("id"), extra_info.get("id"), default="")),
                    "claim": _get_claim(ex),
                    "query": query,
                    "query_len": len(tokenize_for_bm25(query)),
                    "retrieved_doc_ids": retrieved,
                    "metrics": metrics,
                    "reward": reward,
                    "strategy": strategy_name,
                }
            )

        if show_progress and not use_tqdm and (idx % progress_every == 0 or idx == len(data)):
            print(f"[progress] {strategy_name}: {idx}/{len(data)}")

    n = max(len(data), 1)
    summary = {key: value / n for key, value in totals.items()}
    if collect_details:
        return summary, details
    return summary


def evaluate(
    data: List[dict],
    env: ShardedBM25SentenceEnv,
    strategy_fn: Callable[[dict], str],
    runs: int = 1,
    reward_weights: RewardWeights | None = None,
    eval_topk: int = 10,
    *,
    show_progress: bool = True,
    progress_every: int = 10,
    collect_details: bool = False,
) -> Dict[str, tuple[float, float]] | tuple[Dict[str, tuple[float, float]], List[dict]]:
    if collect_details:
        run_summary, details = evaluate_once(
            data,
            env,
            strategy_fn,
            reward_weights=reward_weights,
            eval_topk=eval_topk,
            show_progress=show_progress,
            progress_every=progress_every,
            collect_details=True,
        )
        return summarize_runs([run_summary]), details

    run_rows = [
        evaluate_once(
            data,
            env,
            strategy_fn,
            reward_weights=reward_weights,
            eval_topk=eval_topk,
            show_progress=show_progress,
            progress_every=progress_every,
            collect_details=False,
        )
        for _ in range(runs)
    ]
    return summarize_runs(run_rows)


def load_eval_data(path: str | Path, max_examples: Optional[int] = 1000) -> List[dict]:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        if max_examples is not None:
            df = df.head(max_examples)
        return [_normalise_record(row) for row in df.to_dict(orient="records")]

    data = load_jsonl(path)
    if max_examples is not None:
        data = data[:max_examples]
    return [_normalise_record(row) for row in data]


def save_jsonl(rows: List[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_plain(row), ensure_ascii=False) + "\n")


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant:"


def _load_llm(model_path: str, tokenizer_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
    return model, tokenizer


def make_llm_prompt_query_fn(
    *,
    model_path: str,
    tokenizer_path: str | None = None,
    max_new_tokens: int = 32,
) -> Callable[[dict], str]:
    import torch

    model, tokenizer = _load_llm(model_path=model_path, tokenizer_path=tokenizer_path)

    @torch.inference_mode()
    def llm_prompt_query(ex: dict) -> str:
        claim = _get_claim(ex)
        messages = build_messages(claim)
        prompt_text = _render_prompt(tokenizer, messages)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return parse_model_query_output(
            raw_text,
            claim=claim,
            max_terms=3,
            fallback_to_rule_based=False,
        )

    llm_prompt_query.__name__ = "llm_prompt_query"
    return llm_prompt_query


def default_eval_path(path_cfg: PathConfig) -> Path:
    formal_subset = path_cfg.export_dir / "valid_formal_subset.parquet"
    if formal_subset.exists():
        return formal_subset
    if path_cfg.valid_for_verl_path.exists():
        return path_cfg.valid_for_verl_path
    return path_cfg.dev_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Default: outputs/valid_formal_subset.parquet if it exists, otherwise PathConfig.valid_for_verl_path/dev_path.",
    )
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--eval-topk", type=int, default=10)
    parser.add_argument("--strategy", choices=["all", "claim", "entity", "llm_prompt"], default="all")
    parser.add_argument("--model-path", type=str, default=None, help="Required for --strategy llm_prompt or all.")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Default: --model-path.")
    parser.add_argument("--llm-max-new-tokens", type=int, default=32)
    parser.add_argument("--save-details", action="store_true")
    parser.add_argument("--details-path", type=str, default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="When tqdm is unavailable, print a progress update every N examples.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display.",
    )
    args = parser.parse_args()

    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.llm_max_new_tokens <= 0:
        raise ValueError("--llm-max-new-tokens must be positive")
    if args.strategy in {"all", "llm_prompt"} and not args.model_path:
        raise ValueError("--model-path is required when running the llm_prompt baseline or --strategy all")

    path_cfg = PathConfig()
    ensure_dirs(path_cfg)
    env = require_corpus(path_cfg)
    data_path = Path(args.input_path) if args.input_path else default_eval_path(path_cfg)
    dev_data = load_eval_data(data_path, max_examples=args.max_examples)
    show_progress = not args.no_progress

    strategies: list[tuple[str, Callable[[dict], str]]] = []
    if args.strategy in {"all", "claim"}:
        strategies.append(("BM25 + Claim", claim_only_query))
    if args.strategy in {"all", "entity"}:
        strategies.append(("BM25 + Entity", entity_query))
    if args.strategy in {"all", "llm_prompt"}:
        strategies.append(
            (
                "LLM Prompt + BM25 (Same Prompt, No GRPO)",
                make_llm_prompt_query_fn(
                    model_path=args.model_path,
                    tokenizer_path=args.tokenizer_path,
                    max_new_tokens=args.llm_max_new_tokens,
                ),
            )
        )

    results: dict[str, object] = {
        "reward_formula": REWARD_FORMULA,
        "num_shards": env.num_shards,
        "input_path": str(data_path),
        "max_examples": args.max_examples,
    }
    all_details: List[dict] = []
    details_path = Path(args.details_path) if args.details_path else path_cfg.baseline_eval_details_path
    
    for label, fn in strategies:
        print(f"\n[start] Running baseline: {label}")
        if args.save_details:
            summary, details = evaluate(
                dev_data,
                env,
                fn,
                eval_topk=args.eval_topk,
                show_progress=show_progress,
                progress_every=args.progress_every,
                collect_details=True,
            )
            results[label] = summary
            all_details.extend(details)
            save_jsonl(all_details, details_path)
        else:
            results[label] = evaluate(
                dev_data,
                env,
                fn,
                eval_topk=args.eval_topk,
                show_progress=show_progress,
                progress_every=args.progress_every,
                collect_details=False,
            )
        path_cfg.baseline_eval_results_path.write_text(
            json.dumps(_to_plain(results), indent=2),
            encoding="utf-8",
        )
        print(f"[saved] Current results saved after: {label}")
        
    print(json.dumps(_to_plain(results), indent=2))
    env.close()


if __name__ == "__main__":
    main()
