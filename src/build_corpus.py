"""Pre-build and cache the full BM25 corpus as SQLite + two-stage page/sentence indexes."""

from __future__ import annotations

import argparse
import json

from bm25_env import ShardedBM25SentenceEnv, build_sqlite_sharded_bm25
from config import PathConfig, SHARD_SIZE, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()

    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    path_cfg = PathConfig()
    ensure_dirs(path_cfg)

    if ShardedBM25SentenceEnv.exists(path_cfg) and not args.force_rebuild:
        env = ShardedBM25SentenceEnv.load(path_cfg)
        source = "cache"
        meta = (
            json.loads(path_cfg.corpus_meta_file.read_text(encoding="utf-8"))
            if path_cfg.corpus_meta_file.exists()
            else {}
        )
    else:
        meta = build_sqlite_sharded_bm25(
            wiki_dir=path_cfg.wiki_dir,
            path_cfg=path_cfg,
            shard_size=args.shard_size,
            force_rebuild=args.force_rebuild,
            show_progress=True,
        )
        env = ShardedBM25SentenceEnv.load(path_cfg)
        source = "rebuilt"

    summary = {
        "source": source,
        "sqlite_db_path": str(path_cfg.sqlite_db_path),
        "shard_dir": str(path_cfg.bm25_shard_dir),
        "page_bm25_path": str(path_cfg.page_bm25_path),
        "num_shards": env.num_shards,
        "num_docs": env.num_docs,
        "num_pages": env.num_pages,
        "meta": meta,
    }
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
