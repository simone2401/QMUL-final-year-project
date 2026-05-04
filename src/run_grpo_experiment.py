"""Run verl GRPO in 100-step chunks with external early stopping(close in formal) and best-checkpoint tracking."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from config import PathConfig, ensure_dirs


@dataclass
class StopDecision:
    should_stop: bool
    reason: str = ""


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def summarize_validation_file(path: str | Path) -> dict[str, float]:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"validation file is empty: {path}")

    def mean(key: str) -> float:
        vals = [float(row.get(key, 0.0)) for row in rows]
        return sum(vals) / len(vals)

    return {
        "step": float(rows[0].get("step", Path(path).stem)),
        "reward": mean("score"),
        "sentence_recall@5": mean("sentence_recall@5"),
        "sentence_recall@10": mean("sentence_recall@10"),
        "page_recall@5": mean("page_recall@5"),
        "page_recall@10": mean("page_recall@10"),
        "full_evidence@10": mean("full_evidence@10"),
        "mrr": mean("mrr"),
        "reward_doc": mean("reward_doc"),
        "reward_sent": mean("reward_sent"),
        "reward_full": mean("reward_full"),
        "reward_mrr": mean("reward_mrr"),
        "query_len": mean("query_len"),
        "claim_len": mean("claim_len"),
        "query_growth_ratio": mean("query_growth_ratio"),
    }

#Collect all TensorBoard event files under the experiment directory.
def _collect_scalar_candidates(exp_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for root, _, files in os.walk(exp_dir):
        for name in files:
            if name.startswith("events.out.tfevents"):
                candidates.append(Path(root) / name)
    return sorted(candidates)

#KL is close in formal experiment
def try_read_latest_kl(exp_dir: Path) -> float | None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None

    latest_value: float | None = None
    latest_step = -1
    tags_seen: set[str] = set()
    for event_file in _collect_scalar_candidates(exp_dir):
        try:
            acc = EventAccumulator(str(event_file))
            acc.Reload()
            tags = acc.Tags().get("scalars", [])
            tags_seen.update(tags)
            kl_tags = [
                tag for tag in tags
                if "kl" in tag.lower() and "coef" not in tag.lower() and "loss" not in tag.lower()
            ]
            for tag in kl_tags:
                scalars = acc.Scalars(tag)
                if not scalars:
                    continue
                if scalars[-1].step >= latest_step:
                    latest_step = scalars[-1].step
                    latest_value = float(scalars[-1].value)
        except Exception:
            continue
    return latest_value


def latest_validation_file(validation_dir: Path, expected_step: int) -> Path:
    direct = validation_dir / f"{expected_step}.jsonl"
    if direct.exists():
        return direct
    candidates = sorted(validation_dir.glob("*.jsonl"), key=lambda p: int(p.stem))
    if not candidates:
        raise FileNotFoundError(f"No validation jsonl found under {validation_dir}")
    return candidates[-1]


#Always make link_path point to the latest/optimally selected checkpoint directory
def safe_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target, target_is_directory=True)


#Save a Python object to a JSON file.
def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_history_csv(history: list[dict], path: Path) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    lines = [",".join(keys)]
    for row in history:
        lines.append(",".join(str(row.get(k, "")) for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_chunk(config_path: Path, workdir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "--config-path",
        str(config_path.parent),
        "--config-name",
        config_path.name,
    ]
    subprocess.run(cmd, cwd=str(workdir), check=True)


def decide_stop(
    history: list[dict],
    patience: int,
    min_improve: float,
    latest_kl: float | None,
    kl_threshold: float,
) -> StopDecision:
    if len(history) < 2:
        return StopDecision(False)

    best_so_far = max(row["sentence_recall@10"] for row in history[:-1])
    latest = history[-1]

    no_improve = 0
    running_best = -1.0
    for row in history:
        if row["sentence_recall@10"] > running_best + min_improve:
            running_best = row["sentence_recall@10"]
            no_improve = 0
        else:
            no_improve += 1
    if no_improve >= patience:
        return StopDecision(True, f"Recall@10 has not improved by more than {min_improve:.3f} for {no_improve} consecutive validations")

    if latest_kl is not None and len(history) >= 3:
        if latest_kl > kl_threshold and latest["sentence_recall@10"] < history[-2]["sentence_recall@10"] < history[-3]["sentence_recall@10"]:
            return StopDecision(True, f"KL persistently high （latest {latest_kl:.4f}）and validation Recall@10 continuosly decrease")

    baseline_growth = history[0].get("query_growth_ratio", 1.0)
    if baseline_growth <= 0:
        baseline_growth = 1.0
    if latest["query_growth_ratio"] > baseline_growth * 1.5 and latest["sentence_recall@10"] <= best_so_far + 1e-12:
        return StopDecision(True, "Average query length has increased by more than 50% from the baseline, but the main validation metric has not improved, indicating possible reward hacking")

    return StopDecision(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--stage", required=True, choices=["smoke", "formal"])
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-improve", type=float, default=0.005)
    parser.add_argument("--kl-threshold", type=float, default=0.12)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    workdir = config_path.parent.parent.parent  # outputs/grpo_runs/configs -> project_root-ish when generated by train_grpo.py
    path_cfg = PathConfig()
    ensure_dirs(path_cfg)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    trainer = config.setdefault("trainer", {})
    exp_name = str(trainer.get("experiment_name", f"bm25_query_grpo_{args.stage}"))
    exp_dir = Path(trainer["default_local_dir"])
    validation_dir = Path(trainer["validation_data_dir"])
    metrics_dir = path_cfg.experiment_metrics_dir / exp_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_json = metrics_dir / f"{args.stage}_validation_history.json"
    history_csv = metrics_dir / f"{args.stage}_validation_history.csv"
    summary_json = metrics_dir / f"{args.stage}_run_summary.json"
    best_link = exp_dir / "best_checkpoint"

    history: list[dict] = []
    best_metric = -1.0
    best_step = 0
    stop_reason = "completed"

    steps = list(range(args.chunk_size, args.max_steps + 1, args.chunk_size))
    if steps[-1] != args.max_steps:
        steps.append(args.max_steps)

    for target_step in steps:
        config["trainer"]["total_training_steps"] = int(target_step)
        chunk_cfg = config_path.parent / f"{config_path.stem}.step{target_step}.yaml"
        chunk_cfg.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_chunk(chunk_cfg, workdir=workdir)

        val_file = latest_validation_file(validation_dir, target_step)
        metrics = summarize_validation_file(val_file)
        metrics["kl"] = try_read_latest_kl(exp_dir)
        history.append(metrics)
        dump_json(history, history_json)
        write_history_csv(history, history_csv)

        current_metric = metrics["sentence_recall@10"]
        if current_metric > best_metric + args.min_improve:
            best_metric = current_metric
            best_step = int(metrics["step"])
            ckpt_dir = exp_dir / f"global_step_{best_step}"
            if ckpt_dir.exists():
                safe_symlink(ckpt_dir, best_link)

        if args.stage == "formal":
            decision = decide_stop(
                history=history,
                patience=args.patience,
                min_improve=args.min_improve,
                latest_kl=metrics["kl"],
                kl_threshold=args.kl_threshold,
            )
            if decision.should_stop:
                #stop_reason = decision.reason
                stop_reason = f"would_stop_but_disabled: {decision.reason}"
                #break

    summary = {
        "stage": args.stage,
        "max_steps": args.max_steps,
        "chunk_size": args.chunk_size,
        "best_step": best_step,
        "best_sentence_recall@10": best_metric,
        "stop_reason": stop_reason,
        "history_path": str(history_json),
        "best_checkpoint": str((exp_dir / f"global_step_{best_step}") if best_step else ""),
    }
    dump_json(summary, summary_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
