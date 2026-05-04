"""Reward computation and retrieval metrics for fact-checking."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from config import RewardWeights


def make_doc_id(page_id: str, line_idx: int | str) -> str:
    return f"{page_id}:{int(line_idx)}"


#Canonicalize evidence sets into a clean list of sets
def canonicalize_evidence_sets(evidence_sets: Sequence[Iterable[str]]) -> List[set[str]]:
    groups: List[set[str]] = []
    for group in evidence_sets:
        normalized = {str(x) for x in group if str(x)}
        if normalized:
            groups.append(normalized)
    return groups


def get_gold_doc_ids(example: dict) -> List[str]:
    gold = []
    for group in example.get("evidence", []):
        for ev in group:
            if len(ev) >= 4 and ev[2] is not None and ev[3] is not None:
                gold.append(make_doc_id(ev[2], ev[3]))
    return list(dict.fromkeys(gold))


def get_gold_page_ids(example: dict) -> List[str]:
    gold_pages = []
    for group in example.get("evidence", []):
        for ev in group:
            if len(ev) >= 4 and ev[2] is not None:
                gold_pages.append(str(ev[2]))
    return list(dict.fromkeys(gold_pages))


def get_gold_evidence_sets(example: dict) -> List[set[str]]:
    groups: List[set[str]] = []
    for group in example.get("evidence", []):
        sentence_ids = set()
        for ev in group:
            if len(ev) >= 4 and ev[2] is not None and ev[3] is not None:
                sentence_ids.add(make_doc_id(ev[2], ev[3]))
        if sentence_ids:
            groups.append(sentence_ids)
    return groups


def recall_at_k(retrieved_doc_ids: List[str], gold_doc_ids: List[str], k: int) -> int:
    return int(any(doc_id in retrieved_doc_ids[:k] for doc_id in gold_doc_ids))


def page_recall_at_k(
    retrieved_doc_ids: List[str],
    doc_page_map: Dict[str, str],
    gold_page_ids: List[str],
    k: int,
) -> int:
    retrieved_pages = [doc_page_map[d] for d in retrieved_doc_ids[:k] if d in doc_page_map]
    return int(any(page_id in retrieved_pages for page_id in gold_page_ids))


def full_evidence_at_k(retrieved_doc_ids: List[str], gold_evidence_sets: List[set[str]], k: int) -> int:
    topk = set(retrieved_doc_ids[:k])
    return int(any(group.issubset(topk) for group in gold_evidence_sets if group))


def mrr_score(retrieved_doc_ids: List[str], gold_doc_ids: List[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics_from_gold(
    *,
    retrieved_doc_ids: List[str],
    doc_page_map: Dict[str, str],
    gold_doc_ids: Sequence[str],
    gold_page_ids: Sequence[str],
    gold_evidence_sets: Sequence[Iterable[str]],
) -> Dict[str, float]:
    gold_doc_ids = list(dict.fromkeys(str(x) for x in gold_doc_ids if str(x)))
    gold_page_ids = list(dict.fromkeys(str(x) for x in gold_page_ids if str(x)))
    gold_evidence_sets = canonicalize_evidence_sets(gold_evidence_sets)

    sent_hit_at_5 = float(recall_at_k(retrieved_doc_ids, gold_doc_ids, 5))
    sent_hit_at_10 = float(recall_at_k(retrieved_doc_ids, gold_doc_ids, 10))
    doc_hit_at_5 = float(page_recall_at_k(retrieved_doc_ids, doc_page_map, gold_page_ids, 5))
    doc_hit_at_10 = float(page_recall_at_k(retrieved_doc_ids, doc_page_map, gold_page_ids, 10))
    full_ev_at_10 = float(full_evidence_at_k(retrieved_doc_ids, gold_evidence_sets, 10))
    mrr = float(mrr_score(retrieved_doc_ids, gold_doc_ids))

    return {
        "sentence_recall@5": sent_hit_at_5,
        "sentence_recall@10": sent_hit_at_10,
        "page_recall@5": doc_hit_at_5,
        "page_recall@10": doc_hit_at_10,
        "full_evidence@10": full_ev_at_10,
        "mrr": mrr,
        "sent_hit@5": sent_hit_at_5,
        "sent_hit@10": sent_hit_at_10,
        "doc_hit@5": doc_hit_at_5,
        "doc_hit@10": doc_hit_at_10,
    }


def compute_retrieval_metrics(
    example: dict,
    retrieved_doc_ids: List[str],
    doc_page_map: Dict[str, str],
) -> Dict[str, float]:
    return compute_retrieval_metrics_from_gold(
        retrieved_doc_ids=retrieved_doc_ids,
        doc_page_map=doc_page_map,
        gold_doc_ids=get_gold_doc_ids(example),
        gold_page_ids=get_gold_page_ids(example),
        gold_evidence_sets=get_gold_evidence_sets(example),
    )


def compute_reward(metrics: Dict[str, float], weights: RewardWeights | None = None) -> float:
    w = weights or RewardWeights()
    reward = (
        w.doc_hit_at_5 * metrics.get("doc_hit@5", 0.0) +
        w.sent_hit_at_5 * metrics.get("sent_hit@5", 0.0) +
        w.full_evidence_at_10 * metrics.get("full_evidence@10", 0.0) +
        w.mrr * metrics.get("mrr", 0.0)
    )
    return float(reward)


def summarize_runs(run_metrics: List[Dict[str, float]]) -> Dict[str, tuple[float, float]]:
    if not run_metrics:
        raise ValueError("run_metrics must not be empty")
    keys = list(run_metrics[0].keys())
    out: Dict[str, tuple[float, float]] = {}
    for key in keys:
        vals = [row[key] for row in run_metrics]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        out[key] = (float(mean), float(var ** 0.5))
    return out
