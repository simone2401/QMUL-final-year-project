"""
Custom verl reward wrapper for SQLite-backed sharded BM25 retrieval.

It converts model output into a retrieval query, runs BM25 retrieval,
computes retrieval metrics, and returns the reward used by GRPO.

"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from bm25_env import ShardedBM25SentenceEnv
from config import PathConfig, RewardWeights
from query_parser import parse_model_query_output, tokenize_for_bm25
from reward_fn import compute_retrieval_metrics_from_gold, compute_reward


#Load and cache the BM25 retrieval environment.
@lru_cache(maxsize=1)
def _get_env() -> ShardedBM25SentenceEnv:
    path_cfg = PathConfig()
    return ShardedBM25SentenceEnv.load(path_cfg)


#Compute reward and detailed retrieval diagnostics from model output.
def score_with_details(
    solution_str: str,
    claim: str,
    gold_doc_ids: list[str],
    gold_page_ids: list[str],
    gold_evidence_sets: list[list[str]],
    *,
    topk: int = 10,
    reward_weights: RewardWeights | None = None,
) -> Dict[str, Any]:
    env = _get_env()
    weights = reward_weights or RewardWeights()
    query = parse_model_query_output(
        solution_str,
        claim=claim,
        max_terms=3,
        fallback_to_rule_based=False,
    )
    retrieved_doc_ids, local_page_map = env.retrieve_ids_with_page_map(query, topk=topk)
    metrics = compute_retrieval_metrics_from_gold(
        retrieved_doc_ids=retrieved_doc_ids,
        doc_page_map=local_page_map,
        gold_doc_ids=gold_doc_ids,
        gold_page_ids=gold_page_ids,
        gold_evidence_sets=gold_evidence_sets,
    )
    reward = compute_reward(metrics, weights=weights)

    claim_len = max(len(tokenize_for_bm25(claim)), 1)
    query_len = len(tokenize_for_bm25(query))
    query_growth_ratio = query_len / claim_len

    reward_doc = float(weights.doc_hit_at_5 * metrics.get("doc_hit@5", 0.0))
    reward_sent = float(weights.sent_hit_at_5 * metrics.get("sent_hit@5", 0.0))
    reward_full = float(weights.full_evidence_at_10 * metrics.get("full_evidence@10", 0.0))
    reward_mrr = float(weights.mrr * metrics.get("mrr", 0.0))

    return {
        "query": query,
        "retrieved_doc_ids": retrieved_doc_ids,
        "metrics": metrics,
        "reward": float(reward),
        "reward_doc": reward_doc,
        "reward_sent": reward_sent,
        "reward_full": reward_full,
        "reward_mrr": reward_mrr,
        "query_len": int(query_len),
        "claim_len": int(claim_len),
        "query_growth_ratio": float(query_growth_ratio),
    }

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> dict[str, Any] | float:
    extra_info = extra_info or {}
    claim = extra_info.get("claim") or (ground_truth.get("claim") if isinstance(ground_truth, dict) else ground_truth)
    if not isinstance(claim, str) or not claim.strip():
        return {
            "score": 0.0,
            "sentence_recall@5": 0.0,
            "sentence_recall@10": 0.0,
            "page_recall@5": 0.0,
            "page_recall@10": 0.0,
            "full_evidence@10": 0.0,
            "mrr": 0.0,
            "reward_doc": 0.0,
            "reward_sent": 0.0,
            "reward_full": 0.0,
            "reward_mrr": 0.0,
            "query_len": 0.0,
            "claim_len": 0.0,
            "query_growth_ratio": 0.0,
        }

    gold_doc_ids = list(extra_info.get("gold_doc_ids") or (ground_truth.get("gold_doc_ids") if isinstance(ground_truth, dict) else []) or [])
    gold_page_ids = list(extra_info.get("gold_page_ids") or (ground_truth.get("gold_page_ids") if isinstance(ground_truth, dict) else []) or [])
    gold_evidence_sets = list(extra_info.get("gold_evidence_sets") or (ground_truth.get("gold_evidence_sets") if isinstance(ground_truth, dict) else []) or [])

    details = score_with_details(
        solution_str=solution_str,
        claim=claim,
        gold_doc_ids=gold_doc_ids,
        gold_page_ids=gold_page_ids,
        gold_evidence_sets=gold_evidence_sets,
        topk=10,
    )
    metrics = details["metrics"]

    return {
        "score": float(details["reward"]),
        "sentence_recall@5": float(metrics["sentence_recall@5"]),
        "sentence_recall@10": float(metrics["sentence_recall@10"]),
        "page_recall@5": float(metrics["page_recall@5"]),
        "page_recall@10": float(metrics["page_recall@10"]),
        "full_evidence@10": float(metrics["full_evidence@10"]),
        "mrr": float(metrics["mrr"]),
        "reward_doc": float(details["reward_doc"]),
        "reward_sent": float(details["reward_sent"]),
        "reward_full": float(details["reward_full"]),
        "reward_mrr": float(details["reward_mrr"]),
        "query_len": float(details["query_len"]),
        "claim_len": float(details["claim_len"]),
        "query_growth_ratio": float(details["query_growth_ratio"]),
    }
