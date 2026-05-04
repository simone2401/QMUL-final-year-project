"""
Central configuration for the SQLite-backed BM25 -> verl/GRPO pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
random.seed(SEED)


@dataclass
class PathConfig:
    project_root: Path = PROJECT_ROOT

    data_root: Path = PROJECT_ROOT / "FEVER"
    wiki_dir: Path = data_root / "wiki-pages"
    train_path: Path = data_root / "train.jsonl"
    dev_path: Path = data_root / "shared_task_dev.jsonl"
    test_path: Path = data_root / "shared_task_test.jsonl"

    output_root: Path = PROJECT_ROOT / "outputs"
    cache_dir: Path = PROJECT_ROOT / "cache"
    export_dir: Path = PROJECT_ROOT / "outputs"

    sqlite_db_path: Path = cache_dir / "fever_corpus.db"
    bm25_shard_dir: Path = cache_dir / "bm25_sqlite_shards"
    page_bm25_path: Path = cache_dir / "page_level_bm25.pkl"
    corpus_meta_file: Path = cache_dir / "corpus_meta.json"

    train_for_verl_path: Path = export_dir / "train_for_verl.parquet"
    valid_for_verl_path: Path = export_dir / "valid_for_verl.parquet"
    dataset_summary_path: Path = export_dir / "dataset_summary.json"
    baseline_eval_results_path: Path = export_dir / "baseline_eval_results.json"
    baseline_eval_details_path: Path = export_dir / "baseline_eval_details.jsonl"
    rl_eval_results_path: Path = export_dir / "rl_eval_results.json"
    rl_eval_details_path: Path = export_dir / "rl_eval_details.jsonl"

    verl_config_path: Path = export_dir / "verl_grpo_template.yaml"
    verl_command_path: Path = export_dir / "run_verl_grpo.sh"

    experiment_root: Path = output_root / "grpo_runs"
    experiment_configs_dir: Path = experiment_root / "configs"
    experiment_scripts_dir: Path = experiment_root / "scripts"
    experiment_metrics_dir: Path = experiment_root / "metrics"
    experiment_plots_dir: Path = experiment_root / "plots"
    experiment_tables_dir: Path = experiment_root / "tables"
    experiment_logs_dir: Path = experiment_root / "logs"
    experiment_checkpoints_dir: Path = experiment_root / "checkpoints"
    experiment_val_generations_dir: Path = experiment_root / "validation_generations"
    experiment_rollout_dir: Path = experiment_root / "train_rollouts"


@dataclass
class RunConfig:
    seed: int = SEED
    train_ratio: float = 0.9
    max_examples: int | None = None
    eval_topk: int = 10


@dataclass
class RewardWeights:
    doc_hit_at_5: float = 0.5
    sent_hit_at_5: float = 1.0
    full_evidence_at_10: float = 1.0
    mrr: float = 0.2


SHARD_SIZE = 1000000
BM25_REMOVE_STOPWORDS = True
REWARD_FORMULA = "R = 0.5 * doc_hit@5 + 1.0 * sent_hit@5 + 1.0 * full_evidence@10 + 0.2 * mrr"

# Two-stage retrieval defaults.
DEFAULT_CANDIDATE_PAGES = 10
PAGE_BM25_TITLE_REPEAT = 1
PAGE_BM25_MAX_SENTENCES = 0
PAGE_BM25_MAX_TOKENS = 16
SQLITE_IN_MAX = 900

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "by", "with", "as", "and", "or",
    "from", "that", "this", "these", "those", "it", "its", "into", "about",
    "after", "before", "during", "over", "under", "between", "than", "then",
    "he", "she", "they", "them", "his", "her", "their", "you", "your", "i",
}

RELATION_KEYWORDS = {
    "is", "was", "were", "are", "be", "become", "became", "has", "have", "had",
    "win", "won", "lose", "lost", "play", "played", "born", "died", "married",
    "directed", "stars", "starred", "located", "founded", "released", "published",
    "served", "president", "prime", "minister", "capital", "member", "author",
    "composer", "singer", "actor", "actress", "ceo", "chairman", "queen", "king",
}

SYSTEM_PROMPT = (
    "You help with evidence retrieval for fact-checking.\n"
    "Your task is NOT to rewrite the claim into a new query.\n"
    "Instead, keep the original claim unchanged and add 2 to 3 short, relevant search terms.\n"
    "The added terms should help BM25 retrieval by emphasizing key entities, relations, dates, roles, or important context.\n"
    "Do not explain anything.\n"
    "Do not output a full rewritten sentence.\n"
    "Return only one single line in this format:\n"
    "<original claim> <term1> <term2> <term3>"
)


def build_messages(claim: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Claim: {claim}\n"
                "Keep the claim exactly unchanged, and append 2 to 3 useful search terms only."
            ),
        },
    ]


def ensure_dirs(path_cfg: PathConfig) -> None:
    path_cfg.output_root.mkdir(parents=True, exist_ok=True)
    path_cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    path_cfg.export_dir.mkdir(parents=True, exist_ok=True)
    path_cfg.bm25_shard_dir.mkdir(parents=True, exist_ok=True)

    path_cfg.experiment_root.mkdir(parents=True, exist_ok=True)
    path_cfg.experiment_configs_dir.mkdir(parents=True, exist_ok=True)
    path_cfg.experiment_scripts_dir.mkdir(parents=True, exist_ok=True)
    path_cfg.experiment_metrics_dir.mkdir(parents=True, exist_ok=True)
    path_cfg.experiment_plots_dir.mkdir(parents=True, exist_ok=True)
    path_cfg.experiment_tables_dir.mkdir(parents=True, exist_ok=True)
    path_cfg.experiment_logs_dir.mkdir(parents=True, exist_ok=True)
    path_cfg.experiment_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    path_cfg.experiment_val_generations_dir.mkdir(parents=True, exist_ok=True)
    path_cfg.experiment_rollout_dir.mkdir(parents=True, exist_ok=True)
