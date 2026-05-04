"""
Unified query parsing and baseline query generation helpers.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from config import RELATION_KEYWORDS, STOPWORDS, build_messages

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = (
        text.replace("-LRB-", " ")
        .replace("-RRB-", " ")
        .replace("-LSB-", " ")
        .replace("-RSB-", " ")
        .replace("-LCB-", " ")
        .replace("-RCB-", " ")
        .replace("_", " ")
    )
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_bm25(text: str, remove_stopwords: bool = False) -> List[str]:
    tokens = normalize_text(text).split()
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in STOPWORDS]
    return tokens

#Remove duplicates while preserving original order
def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        norm = normalize_text(item)
        if norm and norm not in seen:
            seen.add(norm)
            output.append(item.strip())
    return output


def extract_query_terms_rule_based(claim: str, max_terms: int = 3) -> List[str]:
    claim = claim.strip()
    if not claim:
        return []

    if _NLP is None:
        quoted = re.findall(r'"([^"]+)"', claim)
        years = re.findall(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b", claim)
        capitalized = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", claim)
        relation_hits = [
            tok for tok in re.findall(r"[A-Za-z]+", claim) if tok.lower() in RELATION_KEYWORDS
        ]
        candidates = dedupe_preserve_order(quoted + capitalized + years + relation_hits)
        return candidates[:max_terms]

    doc = _NLP(claim)
    entity_terms = [ent.text.strip() for ent in doc.ents if ent.text.strip()]
    relation_terms: List[str] = []
    for tok in doc:
        lemma = tok.lemma_.lower().strip()
        if lemma in RELATION_KEYWORDS:
            relation_terms.append(tok.text.strip())

    noun_chunks: List[str] = []
    for chunk in doc.noun_chunks:
        txt = chunk.text.strip()
        if 1 <= len(txt.split()) <= 4:
            noun_chunks.append(txt)

    candidates = dedupe_preserve_order(entity_terms + relation_terms + noun_chunks)
    short_candidates = [c for c in candidates if len(c.split()) <= 4]
    return short_candidates[:max_terms]


def build_query_from_claim_and_terms(claim: str, terms: List[str]) -> str:
    claim = re.sub(r"\s+", " ", claim.strip())
    clean_terms = [re.sub(r"\s+", " ", t.strip()) for t in terms if t and t.strip()]
    clean_terms = dedupe_preserve_order(clean_terms)
    if not clean_terms:
        return claim
    return f"{claim} {' '.join(clean_terms)}".strip()


def clean_model_output(text: str) -> str:
    text = text.strip().replace("\n", " ")
    text = re.sub(r"^assistant\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^query\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_model_query_output(
    raw_text: str,
    claim: str,
    max_terms: int = 3,
    fallback_to_rule_based: bool = False,
) -> str:
    claim = re.sub(r"\s+", " ", claim.strip())
    raw_text = clean_model_output(raw_text)

    if not raw_text:
        terms = extract_query_terms_rule_based(claim, max_terms=max_terms) if fallback_to_rule_based else []
        return build_query_from_claim_and_terms(claim, terms)

    raw_norm = normalize_text(raw_text)
    claim_norm = normalize_text(claim)
    candidate_terms: List[str] = []

    if raw_norm.startswith(claim_norm):
        suffix = raw_text[len(claim):].strip()
        candidate_terms = suffix.split()
    else:
        claim_token_set = set(tokenize_for_bm25(claim))
        raw_tokens = re.findall(r"[A-Za-z0-9\-]+", raw_text)
        candidate_terms = [tok for tok in raw_tokens if normalize_text(tok) not in claim_token_set]

    candidate_terms = dedupe_preserve_order(candidate_terms)
    candidate_terms = [tok for tok in candidate_terms if normalize_text(tok)]
    candidate_terms = candidate_terms[:max_terms]

    if not candidate_terms:
        candidate_terms = []

    return build_query_from_claim_and_terms(claim, candidate_terms)


def claim_only_query(example: dict) -> str:
    return example["claim"]


def entity_query(example: dict, max_terms: int = 3) -> str:
    claim = example["claim"]
    terms = extract_query_terms_rule_based(claim, max_terms=max_terms)
    return build_query_from_claim_and_terms(claim, terms)


def make_messages_for_claim(example: dict) -> list[dict[str, str]]:
    return build_messages(example["claim"])
