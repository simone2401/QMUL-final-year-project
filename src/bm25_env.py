"""
SQLite-backed BM25 retrieval environment for FEVER sentence retrieval.

SQLite + sentence-shard build artifacts+page-level BM25 index+two-stage retrieval path:
1. Retrieve top-N candidate pages with page-level BM25.
2. Fetch only sentences from those pages from SQLite.
3. Build a tiny in-memory BM25 over those sentences and rank locally.

"""

from __future__ import annotations

import gc
import heapq
import json
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence
from urllib.parse import quote

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from config import (
    BM25_REMOVE_STOPWORDS,
    DEFAULT_CANDIDATE_PAGES,
    PAGE_BM25_MAX_SENTENCES,
    PAGE_BM25_MAX_TOKENS,
    PAGE_BM25_TITLE_REPEAT,
    # Maximum batch size for SQLite IN queries to avoid too many SQL parameters
    SQLITE_IN_MAX,
    PathConfig,
    RewardWeights,
    SHARD_SIZE,
    ensure_dirs,
)
from query_parser import tokenize_for_bm25
from reward_fn import compute_retrieval_metrics, compute_retrieval_metrics_from_gold, compute_reward


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def iter_wiki_pages(wiki_dir: str | Path) -> Generator[dict, None, None]:
    wiki_dir = Path(wiki_dir)
    for file_path in sorted(wiki_dir.glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def make_doc_id(page_id: str, line_idx: int | str) -> str:
    return f"{page_id}:{int(line_idx)}"


def parse_doc_id(doc_id: str) -> tuple[str, int]:
    page_id, line_idx = str(doc_id).rsplit(":", 1)
    return page_id, int(line_idx)


def _page_title(page_id: str) -> str:
    return str(page_id).replace("_", " ")


def _parse_page_lines(page: dict) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for line in page.get("lines", "").split("\n"):
        if "\t" not in line:
            continue
        try:
            sid, text = line.split("\t", 1)
        except ValueError:
            continue
        sid, text = sid.strip(), text.strip()
        if not sid.isdigit() or not text:
            continue
        rows.append((int(sid), text))
    return rows


def _build_page_tokens(page_id: str, sentence_rows: Sequence[tuple[int, str]]) -> list[str]:
    title = page_id.replace("_", " ")
    combined_text = title
    tokens = tokenize_for_bm25(combined_text, remove_stopwords=BM25_REMOVE_STOPWORDS)
    if PAGE_BM25_MAX_TOKENS > 0:
        tokens = tokens[:PAGE_BM25_MAX_TOKENS]
    return tokens


class FeverSQLiteCorpus:
    """Sentence corpus stored in SQLite."""

    def __init__(self, db_path: str | Path, *, read_only: bool = False) -> None:
        self.db_path = Path(db_path)
        self.read_only = read_only
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn

        if self.read_only:
            uri = f"file:{quote(str(self.db_path))}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA mmap_size=268435456;")
        self._conn = conn
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._connect()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def initialize(self, *, drop_existing: bool = False) -> None:
        conn = self.conn
        if drop_existing:
            conn.execute("DROP TABLE IF EXISTS sentences")
            conn.execute("DROP TABLE IF EXISTS metadata")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sentences (
                page_id TEXT NOT NULL,
                line_idx INTEGER NOT NULL,
                sentence_text TEXT NOT NULL,
                PRIMARY KEY (page_id, line_idx)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sentences_page_line ON sentences (page_id, line_idx)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sentences_page_id ON sentences (page_id)"
        )
        conn.commit()

    def set_metadata(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, value),
        )

    def get_metadata(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else default

    def add_sentences(self, rows: Iterable[tuple[str, int, str]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO sentences(page_id, line_idx, sentence_text) VALUES (?, ?, ?)",
            rows,
        )

    def commit(self) -> None:
        self.conn.commit()


@dataclass
class RetrievalResult:
    doc_id: str
    page_id: str
    score: float
    sentence_text: Optional[str] = None


class BM25SentenceShard:
    """Single shard that stores only light retrieval structures."""

    def __init__(
        self,
        bm25: BM25Okapi,
        doc_ids: List[str],
        tokenized_corpus: Optional[List[List[str]]] = None,
    ) -> None:
        self.bm25 = bm25
        self.doc_ids = doc_ids
        self.tokenized_corpus = tokenized_corpus

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[tuple[str, List[str]]],
        *,
        keep_tokenized_corpus: bool = False,
    ) -> "BM25SentenceShard":
        tokenized_corpus: List[List[str]] = []
        doc_ids: List[str] = []

        for doc_id, tokenized in documents:
            if not tokenized:
                continue
            doc_ids.append(str(doc_id))
            tokenized_corpus.append(list(tokenized))

        bm25 = BM25Okapi(tokenized_corpus)
        return cls(
            bm25=bm25,
            doc_ids=doc_ids,
            tokenized_corpus=tokenized_corpus if keep_tokenized_corpus else None,
        )

    def save(self, file_path: str | Path, *, include_tokenized_corpus: bool = False) -> None:
        payload = {
            "bm25": self.bm25,
            "doc_ids": self.doc_ids,
        }
        if include_tokenized_corpus and self.tokenized_corpus is not None:
            payload["tokenized_corpus"] = self.tokenized_corpus

        with Path(file_path).open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: str | Path) -> "BM25SentenceShard":
        with Path(file_path).open("rb") as f:
            payload = pickle.load(f)
        return cls(
            bm25=payload["bm25"],
            doc_ids=payload["doc_ids"],
            tokenized_corpus=payload.get("tokenized_corpus"),
        )

    def retrieve(self, query_tokens: List[str], topk: int = 10) -> List[tuple[str, float]]:
        if not query_tokens or topk <= 0:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = heapq.nlargest(topk, range(len(scores)), key=lambda i: scores[i])
        return [
            (self.doc_ids[i], float(scores[i]))
            for i in top_indices
            if float(scores[i]) != float("-inf")
        ]

    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)


class PageBM25Index:
    """Compact page-level BM25 index used for Stage-1 retrieval."""

    def __init__(self, bm25: BM25Okapi, page_ids: List[str]) -> None:
        self.bm25 = bm25
        self.page_ids = page_ids

    @classmethod
    def from_documents(cls, documents: Iterable[tuple[str, List[str]]]) -> "PageBM25Index":
        tokenized_corpus: list[list[str]] = []
        page_ids: list[str] = []
        for page_id, tokens in documents:
            if not tokens:
                continue
            page_ids.append(str(page_id))
            tokenized_corpus.append(list(tokens))
        return cls(BM25Okapi(tokenized_corpus), page_ids)

    def save(self, file_path: str | Path) -> None:
        with Path(file_path).open("wb") as f:
            pickle.dump({"bm25": self.bm25, "page_ids": self.page_ids}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: str | Path) -> "PageBM25Index":
        with Path(file_path).open("rb") as f:
            payload = pickle.load(f)
        return cls(bm25=payload["bm25"], page_ids=payload["page_ids"])

    def retrieve(self, query_tokens: List[str], topk: int) -> List[tuple[str, float]]:
        if not query_tokens or topk <= 0:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_indices = heapq.nlargest(topk, range(len(scores)), key=lambda i: scores[i])
        return [
            (self.page_ids[i], float(scores[i]))
            for i in top_indices
            if float(scores[i]) != float("-inf")
        ]

    @property
    def num_docs(self) -> int:
        return len(self.page_ids)


class ShardedBM25SentenceEnv:
    """Sentence retrieval environment backed by SQLite plus page-level BM25."""

    def __init__(
        self,
        shard_paths: List[Path],
        sqlite_db_path: str | Path,
        path_cfg: PathConfig | None = None,
        page_bm25_path: str | Path | None = None,
        candidate_pages: int = DEFAULT_CANDIDATE_PAGES,
    ) -> None:
        self.shard_paths = [Path(p) for p in shard_paths]
        self.sqlite_db_path = Path(sqlite_db_path)
        self.path_cfg = path_cfg or PathConfig()
        self.page_bm25_path = Path(page_bm25_path) if page_bm25_path is not None else self.path_cfg.page_bm25_path
        self.candidate_pages = max(int(candidate_pages), 1)
        self._db_conn: sqlite3.Connection | None = None
        self._page_index: PageBM25Index | None = None

    @classmethod
    def shard_paths_from_cfg(cls, path_cfg: PathConfig | None = None) -> List[Path]:
        path_cfg = path_cfg or PathConfig()
        return sorted(path_cfg.bm25_shard_dir.glob("bm25_shard_*.pkl"))

    @classmethod
    def exists(cls, path_cfg: PathConfig | None = None) -> bool:
        path_cfg = path_cfg or PathConfig()
        has_sentence_index = len(cls.shard_paths_from_cfg(path_cfg)) > 0
        has_page_index = path_cfg.page_bm25_path.exists()
        return path_cfg.sqlite_db_path.exists() and (has_page_index or has_sentence_index)

    @classmethod
    def load(cls, path_cfg: PathConfig | None = None) -> "ShardedBM25SentenceEnv":
        path_cfg = path_cfg or PathConfig()
        shard_paths = cls.shard_paths_from_cfg(path_cfg)
        if not path_cfg.sqlite_db_path.exists():
            raise FileNotFoundError(
                f"Missing SQLite corpus at {path_cfg.sqlite_db_path}. Run build_corpus.py first."
            )
        if not shard_paths and not path_cfg.page_bm25_path.exists():
            raise FileNotFoundError(
                f"Missing retrieval indexes under {path_cfg.cache_dir}. Run build_corpus.py first."
            )
        return cls(
            shard_paths=shard_paths,
            sqlite_db_path=path_cfg.sqlite_db_path,
            path_cfg=path_cfg,
            page_bm25_path=path_cfg.page_bm25_path,
        )

    @classmethod
    def load_or_build(
        cls,
        wiki_dir: str | Path,
        path_cfg: PathConfig | None = None,
        *,
        force_rebuild: bool = False,
        shard_size: int = SHARD_SIZE,
        show_progress: bool = True,
    ) -> "ShardedBM25SentenceEnv":
        path_cfg = path_cfg or PathConfig()
        ensure_dirs(path_cfg)
        if cls.exists(path_cfg) and not force_rebuild:
            return cls.load(path_cfg)
        build_sqlite_sharded_bm25(
            wiki_dir=wiki_dir,
            path_cfg=path_cfg,
            shard_size=shard_size,
            force_rebuild=force_rebuild,
            show_progress=show_progress,
        )
        return cls.load(path_cfg)

    def _get_db_conn(self) -> sqlite3.Connection:
        if self._db_conn is None:
            db_uri = f"file:{quote(str(self.sqlite_db_path))}?mode=ro"
            self._db_conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
            self._db_conn.row_factory = sqlite3.Row
            self._db_conn.execute("PRAGMA temp_store=MEMORY;")
            self._db_conn.execute("PRAGMA mmap_size=268435456;")
            self._db_conn.execute("PRAGMA query_only=ON;")
        return self._db_conn

    def _get_page_index(self) -> PageBM25Index | None:
        if self._page_index is None and self.page_bm25_path.exists():
            self._page_index = PageBM25Index.load(self.page_bm25_path)
        return self._page_index

    def close(self) -> None:
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None
        self._page_index = None

    def get_doc_text(self, doc_id: str) -> str:
        page_id, line_idx = parse_doc_id(doc_id)
        row = self._get_db_conn().execute(
            "SELECT sentence_text FROM sentences WHERE page_id = ? AND line_idx = ?",
            (page_id, line_idx),
        ).fetchone()
        return "" if row is None else str(row["sentence_text"])

    def get_doc_texts(self, doc_ids: Sequence[str]) -> Dict[str, str]:
        if not doc_ids:
            return {}

        grouped: dict[str, list[int]] = {}
        for doc_id in doc_ids:
            page_id, line_idx = parse_doc_id(doc_id)
            grouped.setdefault(page_id, []).append(line_idx)

        out: Dict[str, str] = {}
        conn = self._get_db_conn()
        for page_id, line_indices in grouped.items():
            placeholders = ",".join("?" for _ in line_indices)
            sql = (
                f"SELECT page_id, line_idx, sentence_text FROM sentences "
                f"WHERE page_id = ? AND line_idx IN ({placeholders})"
            )
            params = [page_id, *line_indices]
            for row in conn.execute(sql, params):
                out[make_doc_id(str(row["page_id"]), int(row["line_idx"]))] = str(row["sentence_text"])
        return out

    def get_sentences_by_page_ids(self, page_ids: Sequence[str]) -> Dict[str, List[tuple[str, str]]]:
        page_ids = [str(p) for p in page_ids if str(p)]
        if not page_ids:
            return {}

        conn = self._get_db_conn()
        out: Dict[str, List[tuple[str, str]]] = {page_id: [] for page_id in page_ids}
        for start in range(0, len(page_ids), SQLITE_IN_MAX):
            chunk = page_ids[start : start + SQLITE_IN_MAX]
            placeholders = ",".join("?" for _ in chunk)
            sql = (
                "SELECT page_id, line_idx, sentence_text FROM sentences "
                f"WHERE page_id IN ({placeholders}) ORDER BY page_id, line_idx"
            )
            for row in conn.execute(sql, chunk):
                page_id = str(row["page_id"])
                doc_id = make_doc_id(page_id, int(row["line_idx"]))
                out.setdefault(page_id, []).append((doc_id, str(row["sentence_text"])))
        return out

    def _iter_loaded_shards(self) -> Generator[BM25SentenceShard, None, None]:
        for shard_path in self.shard_paths:
            shard = BM25SentenceShard.load(shard_path)
            try:
                yield shard
            finally:
                del shard
                gc.collect()

    def _make_results(
        self,
        merged: Sequence[tuple[str, float]],
        *,
        include_text: bool = False,
    ) -> List[RetrievalResult]:
        doc_ids = [doc_id for doc_id, _ in merged]
        text_map = self.get_doc_texts(doc_ids) if include_text else {}
        out: List[RetrievalResult] = []
        for doc_id, score in merged:
            page_id, _ = parse_doc_id(doc_id)
            out.append(
                RetrievalResult(
                    doc_id=doc_id,
                    page_id=page_id,
                    score=float(score),
                    sentence_text=text_map.get(doc_id),
                )
            )
        return out

    def retrieve_pages(self, query: str, topn_pages: int | None = None) -> List[tuple[str, float]]:
        page_index = self._get_page_index()
        topn_pages = max(int(topn_pages or self.candidate_pages), 1)
        query_tokens = tokenize_for_bm25(query, remove_stopwords=BM25_REMOVE_STOPWORDS)
        if not query_tokens or page_index is None:
            return []
        return page_index.retrieve(query_tokens, topk=topn_pages)

    def _retrieve_legacy(self, query_tokens: List[str], topk: int, *, include_text: bool) -> List[RetrievalResult]:
        merged: List[tuple[str, float]] = []
        for shard in self._iter_loaded_shards():
            merged.extend(shard.retrieve(query_tokens, topk=topk))

        merged.sort(key=lambda x: x[1], reverse=True)

        seen = set()
        deduped: List[tuple[str, float]] = []
        for doc_id, score in merged:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            deduped.append((doc_id, score))
            if len(deduped) >= topk:
                break
        return self._make_results(deduped, include_text=include_text)

    def _retrieve_two_stage(
        self,
        query: str,
        query_tokens: List[str],
        topk: int,
        *,
        include_text: bool,
        topn_pages: int | None = None,
    ) -> List[RetrievalResult]:
        candidate_pages = [page_id for page_id, _ in self.retrieve_pages(query, topn_pages=topn_pages)]
        if not candidate_pages:
            return []

        sentence_rows = self.get_sentences_by_page_ids(candidate_pages)
        local_docs: list[tuple[str, list[str]]] = []
        sentence_text_map: dict[str, str] = {}
        for page_id in candidate_pages:
            title = _page_title(page_id)
            for doc_id, sentence_text in sentence_rows.get(page_id, []):
                tokenized = tokenize_for_bm25(
                    f"{title} {sentence_text}",
                    remove_stopwords=BM25_REMOVE_STOPWORDS,
                )
                if not tokenized:
                    continue
                local_docs.append((doc_id, tokenized))
                sentence_text_map[doc_id] = sentence_text

        if not local_docs:
            return []

        local_shard = BM25SentenceShard.from_documents(local_docs, keep_tokenized_corpus=False)
        merged = local_shard.retrieve(query_tokens, topk=topk)
        results: list[RetrievalResult] = []
        for doc_id, score in merged:
            page_id, _ = parse_doc_id(doc_id)
            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    page_id=page_id,
                    score=float(score),
                    sentence_text=sentence_text_map.get(doc_id) if include_text else None,
                )
            )
        return results

    def retrieve(
        self,
        query: str,
        topk: int = 10,
        *,
        include_text: bool = False,
        topn_pages: int | None = None,
        use_two_stage: bool = True,
    ) -> List[RetrievalResult]:
        if topk <= 0:
            return []

        query_tokens = tokenize_for_bm25(query, remove_stopwords=BM25_REMOVE_STOPWORDS)
        if not query_tokens:
            return []

        if use_two_stage and self._get_page_index() is not None:
            results = self._retrieve_two_stage(
                query,
                query_tokens,
                topk,
                include_text=include_text,
                topn_pages=topn_pages,
            )
            if results:
                return results

        return self._retrieve_legacy(query_tokens, topk, include_text=include_text)

    def retrieve_ids(self, query: str, topk: int = 10) -> List[str]:
        return [r.doc_id for r in self.retrieve(query, topk=topk, include_text=False)]

    def retrieve_ids_with_page_map(self, query: str, topk: int = 10) -> tuple[List[str], Dict[str, str]]:
        results = self.retrieve(query, topk=topk, include_text=False)
        retrieved_ids = [r.doc_id for r in results]
        local_page_map = {r.doc_id: r.page_id for r in results}
        return retrieved_ids, local_page_map

    def retrieve_with_text(self, query: str, topk: int = 10) -> List[RetrievalResult]:
        return self.retrieve(query, topk=topk, include_text=True)

    def step(
        self,
        example: dict,
        query: str,
        topk: int = 10,
        reward_weights: RewardWeights | None = None,
    ) -> dict:
        retrieved_ids, local_page_map = self.retrieve_ids_with_page_map(query=query, topk=topk)
        metrics = compute_retrieval_metrics(
            example=example,
            retrieved_doc_ids=retrieved_ids,
            doc_page_map=local_page_map,
        )
        reward = compute_reward(metrics, weights=reward_weights)
        return {
            "query": query,
            "retrieved_doc_ids": retrieved_ids,
            "metrics": metrics,
            "reward": reward,
            "topk": topk,
        }

    def score_from_gold(
        self,
        *,
        query: str,
        gold_doc_ids: Sequence[str],
        gold_page_ids: Sequence[str],
        gold_evidence_sets: Sequence[Iterable[str]],
        topk: int = 10,
        reward_weights: RewardWeights | None = None,
    ) -> dict:
        retrieved_ids, local_page_map = self.retrieve_ids_with_page_map(query=query, topk=topk)
        metrics = compute_retrieval_metrics_from_gold(
            retrieved_doc_ids=retrieved_ids,
            doc_page_map=local_page_map,
            gold_doc_ids=gold_doc_ids,
            gold_page_ids=gold_page_ids,
            gold_evidence_sets=gold_evidence_sets,
        )
        reward = compute_reward(metrics, weights=reward_weights)
        return {
            "query": query,
            "retrieved_doc_ids": retrieved_ids,
            "metrics": metrics,
            "reward": reward,
            "topk": topk,
        }

    @property
    def num_shards(self) -> int:
        return len(self.shard_paths)

    @property
    def num_docs(self) -> int:
        meta_path = self.path_cfg.corpus_meta_file
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return int(meta.get("num_sentences", 0))
        return 0

    @property
    def num_pages(self) -> int:
        meta_path = self.path_cfg.corpus_meta_file
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return int(meta.get("num_pages", 0))
        return 0


def build_sqlite_sharded_bm25(
    wiki_dir: str | Path,
    path_cfg: PathConfig | None = None,
    *,
    shard_size: int = SHARD_SIZE,
    force_rebuild: bool = False,
    show_progress: bool = True,
) -> dict:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    path_cfg = path_cfg or PathConfig()
    ensure_dirs(path_cfg)
    shard_dir = path_cfg.bm25_shard_dir
    shard_dir.mkdir(parents=True, exist_ok=True)

    if force_rebuild:
        for old in shard_dir.glob("bm25_shard_*.pkl"):
            old.unlink()
        if path_cfg.corpus_meta_file.exists():
            path_cfg.corpus_meta_file.unlink()
        if path_cfg.sqlite_db_path.exists():
            path_cfg.sqlite_db_path.unlink()
        if path_cfg.page_bm25_path.exists():
            path_cfg.page_bm25_path.unlink()

    sqlite_corpus = FeverSQLiteCorpus(path_cfg.sqlite_db_path, read_only=False)
    sqlite_corpus.initialize(drop_existing=force_rebuild)

    shard_buffer: List[tuple[str, List[str]]] = []
    sqlite_buffer: List[tuple[str, int, str]] = []
    page_docs: List[tuple[str, List[str]]] = []
    shard_paths: List[str] = []
    total_docs = 0
    total_pages = 0
    shard_index = 0

    iterator = iter_wiki_pages(wiki_dir)
    if show_progress:
        iterator = tqdm(iterator, desc="Building SQLite + page BM25 corpus", unit="page")

    #Flush buffered sentences into SQLite database in batch
    def flush_sqlite(buffer: List[tuple[str, int, str]]) -> None:
        if buffer:
            sqlite_corpus.add_sentences(buffer)
            sqlite_corpus.commit()

    #Build a BM25 shard from buffered tokenized documents and save it
    def flush_shard(buffer: List[tuple[str, List[str]]], idx: int) -> int:
        nonlocal shard_paths
        if not buffer:
            return 0
        shard = BM25SentenceShard.from_documents(buffer, keep_tokenized_corpus=False)
        shard_path = shard_dir / f"bm25_shard_{idx:04d}.pkl"
        shard.save(shard_path, include_tokenized_corpus=False)
        count = shard.num_docs
        shard_paths.append(str(shard_path))
        del shard
        gc.collect()
        return count

    for page in iterator:
        page_id = str(page["id"])
        sentence_rows = _parse_page_lines(page)
        if not sentence_rows:
            continue

        total_pages += 1
        page_tokens = _build_page_tokens(page_id, sentence_rows)
        if page_tokens:
            page_docs.append((page_id, page_tokens))

        title = _page_title(page_id)
        for line_idx, sentence_text in sentence_rows:
            doc_id = make_doc_id(page_id, line_idx)
            sqlite_buffer.append((page_id, line_idx, sentence_text))
            tokenized = tokenize_for_bm25(
                f"{title} {sentence_text}",
                remove_stopwords=BM25_REMOVE_STOPWORDS,
            )
            if tokenized:
                shard_buffer.append((doc_id, tokenized))

        if len(sqlite_buffer) >= 50000:
            flush_sqlite(sqlite_buffer)
            sqlite_buffer.clear()

        if len(shard_buffer) >= shard_size:
            total_docs += flush_shard(shard_buffer, shard_index)
            shard_index += 1
            shard_buffer.clear()
            gc.collect()

    if sqlite_buffer:
        flush_sqlite(sqlite_buffer)
        sqlite_buffer.clear()

    if shard_buffer:
        total_docs += flush_shard(shard_buffer, shard_index)
        shard_buffer.clear()
        gc.collect()

    print("Building page-level BM25 index...")
    page_index = PageBM25Index.from_documents(page_docs)
    page_index.save(path_cfg.page_bm25_path)
    del page_index
    gc.collect()

    print("Ensuring SQLite index for fast retrieval...")
    sqlite_corpus.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sentences_page_line ON sentences (page_id, line_idx)"
    )
    sqlite_corpus.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sentences_page_id ON sentences (page_id)"
    )

    print("Running SQLite ANALYZE...")
    sqlite_corpus.conn.execute("ANALYZE;")

    sqlite_corpus.set_metadata("format", "sqlite_sentence_corpus")
    sqlite_corpus.set_metadata("remove_stopwords", json.dumps(BM25_REMOVE_STOPWORDS))
    sqlite_corpus.set_metadata("num_pages", str(total_pages))
    sqlite_corpus.set_metadata("num_sentences", str(total_docs))
    sqlite_corpus.set_metadata("page_bm25_path", str(path_cfg.page_bm25_path))
    sqlite_corpus.commit()
    sqlite_corpus.close()

    meta = {
        "format": "sqlite_sharded_bm25_sentence_env_two_stage",
        "sqlite_db_path": str(path_cfg.sqlite_db_path),
        "page_bm25_path": str(path_cfg.page_bm25_path),
        "shard_size": shard_size,
        "num_shards": len(shard_paths),
        "num_sentences": total_docs,
        "num_pages": total_pages,
        "shard_paths": shard_paths,
        "stores_doc_text": False,
        "doc_id_format": "page_id:line_idx",
        "remove_stopwords": BM25_REMOVE_STOPWORDS,
        "retrieval_mode": "two_stage_page_then_sentence",
        "candidate_pages_default": DEFAULT_CANDIDATE_PAGES,
        "page_bm25": {
            "title_repeat": PAGE_BM25_TITLE_REPEAT,
            "max_sentences": PAGE_BM25_MAX_SENTENCES,
            "max_tokens": PAGE_BM25_MAX_TOKENS,
            "num_pages_indexed": len(page_docs),
        },
    }
    path_cfg.corpus_meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
