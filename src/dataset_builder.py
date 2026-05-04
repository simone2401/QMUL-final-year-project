"""Dataset construction, filtering, splitting, and export helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence

from config import PathConfig, RunConfig, build_messages, ensure_dirs
from reward_fn import get_gold_doc_ids, get_gold_evidence_sets, get_gold_page_ids


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def has_valid_evidence(example: dict) -> bool:
    return bool(get_gold_doc_ids(example))


def filter_examples(data: Sequence[dict], require_evidence: bool = True, drop_nei: bool = False) -> List[dict]:
    filtered = []
    for ex in data:
        if require_evidence and not has_valid_evidence(ex):
            continue
        if drop_nei and ex.get("label") == "NOT ENOUGH INFO":
            continue
        filtered.append(ex)
    return filtered


def build_training_sample(example: dict) -> dict:
    claim = example["claim"]
    return {
        "id": example.get("id"),
        "claim": claim,
        "label": example.get("label"),
        "prompt": build_messages(claim),
        "gold_doc_ids": get_gold_doc_ids(example),
        "gold_page_ids": get_gold_page_ids(example),
        "gold_evidence_sets": [sorted(list(x)) for x in get_gold_evidence_sets(example)],
        "meta": {"verifiable": has_valid_evidence(example), "source": "fever"},
    }


def build_dataset(raw_data: Sequence[dict], require_evidence: bool = True, drop_nei: bool = False) -> List[dict]:
    filtered = filter_examples(raw_data, require_evidence=require_evidence, drop_nei=drop_nei)
    return [build_training_sample(ex) for ex in filtered]


def split_dataset(data: Sequence[dict], train_ratio: float = 0.9, seed: int = 42) -> tuple[List[dict], List[dict]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    set_seed(seed)
    data = list(data)
    random.shuffle(data)
    cut = int(len(data) * train_ratio)
    return data[:cut], data[cut:]

def export_for_verl_grpo(dataset: Sequence[dict], output_path: str | Path) -> None:
    from pathlib import Path
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ex in dataset:
        ground_truth = {
            "claim": ex["claim"],
            "gold_doc_ids": ex["gold_doc_ids"],
            "gold_page_ids": ex["gold_page_ids"],
            "gold_evidence_sets": ex["gold_evidence_sets"],
        }
        rows.append({
            "data_source": "fever",
            "prompt": ex["prompt"],
            "claim": ex["claim"],
            "ground_truth": ground_truth,
            "ability": "fact_check_retrieval",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "id": ex["id"],
                "label": ex["label"],
                "claim": ex["claim"],
                "gold_doc_ids": ex["gold_doc_ids"],
                "gold_page_ids": ex["gold_page_ids"],
                "gold_evidence_sets": ex["gold_evidence_sets"],
            },
        })

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    

def build_and_export_default(path_cfg: PathConfig | None = None, run_cfg: RunConfig | None = None, train_ratio: float | None = None, drop_nei: bool = False) -> dict:
    path_cfg = path_cfg or PathConfig()
    run_cfg = run_cfg or RunConfig()
    ensure_dirs(path_cfg)
    raw_train = load_jsonl(path_cfg.train_path)
    dataset = build_dataset(raw_train, require_evidence=True, drop_nei=drop_nei)
    ratio = train_ratio if train_ratio is not None else run_cfg.train_ratio
    train_set, valid_set = split_dataset(dataset, train_ratio=ratio, seed=run_cfg.seed)
    export_for_verl_grpo(train_set, path_cfg.train_for_verl_path)
    export_for_verl_grpo(valid_set, path_cfg.valid_for_verl_path)
    summary = {
        "train_examples": len(train_set),
        "valid_examples": len(valid_set),
        "train_path": str(path_cfg.train_for_verl_path),
        "valid_path": str(path_cfg.valid_for_verl_path),
    }
    path_cfg.dataset_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
