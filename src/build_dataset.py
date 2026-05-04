"""Generate the verl training/validation JSONL files once and reuse them."""

from __future__ import annotations

import argparse
import json

from config import PathConfig, RunConfig, ensure_dirs
from dataset_builder import build_and_export_default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--drop-nei", action="store_true")
    args = parser.parse_args()

    path_cfg = PathConfig()
    ensure_dirs(path_cfg)
    summary = build_and_export_default(
        path_cfg=path_cfg,
        run_cfg=RunConfig(train_ratio=args.train_ratio),
        train_ratio=args.train_ratio,
        drop_nei=args.drop_nei,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
