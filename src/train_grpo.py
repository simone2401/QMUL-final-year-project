"""
Prepare a verl GRPO setup tailored for a single NVIDIA A40 46GB GPU.
Compatible with local verl versions that expect top-level ray_init and a critic node.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

from bm25_env import ShardedBM25SentenceEnv
from config import PathConfig, RunConfig, ensure_dirs
from dataset_builder import build_and_export_default


SMOKE_STEPS = 2
SMOKE_VALIDATE_EVERY = 1

FORMAL_STEPS = 200
FORMAL_VALIDATE_EVERY = 10

DEFAULT_LORA_RANK = 32
DEFAULT_RAY_NUM_CPUS = 8
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.2
DEFAULT_ROLLOUT_LOG_EVERY_N_STEPS = 50
DEFAULT_FORMAL_VALID_EXAMPLES = 1000
DEFAULT_FORMAL_VALID_SEED = 108


def _resolve_existing_dir(path_str: str, arg_name: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{arg_name} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{arg_name} must be a directory: {path}")
    return path


def build_subset_parquet(
    *,
    full_path: Path,
    subset_path: Path,
    subset_examples: int,
    seed: int = 42,
    shuffle: bool = True,
) -> dict:
    df = pd.read_parquet(full_path)
    subset_n = min(int(subset_examples), len(df))
    if subset_n <= 0:
        raise ValueError("subset_examples must be positive and source dataset must be non-empty")

    if shuffle:
        subset_df = df.sample(n=subset_n, random_state=seed).reset_index(drop=True)
    else:
        subset_df = df.head(subset_n).copy()

    subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_parquet(subset_path, index=False)

    return {
        "examples": int(subset_n),
        "path": str(subset_path),
        "seed": int(seed),
        "shuffle": bool(shuffle),
    }


def build_smoke_parquet(
    *,
    full_train_path: Path,
    full_valid_path: Path,
    smoke_train_path: Path,
    smoke_valid_path: Path,
    smoke_train_examples: int,
    smoke_valid_examples: int,
) -> dict:
    smoke_train = build_subset_parquet(
        full_path=full_train_path,
        subset_path=smoke_train_path,
        subset_examples=smoke_train_examples,
        seed=42,
        shuffle=False,
    )
    smoke_valid = build_subset_parquet(
        full_path=full_valid_path,
        subset_path=smoke_valid_path,
        subset_examples=smoke_valid_examples,
        seed=42,
        shuffle=False,
    )
    return {
        "smoke_train_examples": smoke_train["examples"],
        "smoke_valid_examples": smoke_valid["examples"],
        "smoke_train_path": smoke_train["path"],
        "smoke_valid_path": smoke_valid["path"],
    }


def render_yaml(
    *,
    train_file: str,
    val_file: str,
    path_cfg: PathConfig,
    model_path: str,
    tokenizer_path: str,
    experiment_name: str,
    train_batch_size: int,
    val_batch_size: int,
    rollout_n: int,
    ppo_micro_batch_size_per_gpu: int,
    log_prob_micro_batch_size_per_gpu: int,
    total_training_steps: int,
    validate_every: int,
    learning_rate: float,
    lora_rank: int,
    ray_num_cpus: int,
    rollout_gpu_memory_utilization: float,
    rollout_log_every_n_steps: int,
) -> str:
    reward_path = Path(__file__).resolve().parent / "reward.py"
    checkpoint_dir = path_cfg.experiment_checkpoints_dir / experiment_name
    validation_dir = path_cfg.experiment_val_generations_dir / experiment_name
    rollout_dir = path_cfg.experiment_rollout_dir / experiment_name
    return f"""ray_init:
  num_cpus: {ray_num_cpus}
  ignore_reinit_error: True
  include_dashboard: False

data:
  train_files: ['{train_file}']
  val_files: ['{val_file}']
  prompt_key: prompt
  reward_model_key: ground_truth
  data_source_key: data_source
  reward_fn_key: data_source
  tokenizer: {tokenizer_path}
  train_batch_size: {train_batch_size}
  val_batch_size: {val_batch_size}
  max_prompt_length: 512
  max_response_length: 32
  return_raw_chat: True
  shuffle: True
  validation_shuffle: False
  seed: 42

reward_model:
  enable: False
  launch_reward_fn_async: False

custom_reward_function:
  path: {reward_path}
  name: compute_score

algorithm:
  adv_estimator: grpo
  gamma: 1.0
  lam: 1.0
  kl_penalty: low_var_kl
  use_kl_in_reward: False
  use_pf_ppo: False
  pf_ppo:
    reweight_method: "none"
    weight_pow: 1.0

actor_rollout_ref:
  hybrid_engine: True
  model:
    path: {model_path}
    tokenizer_path: {tokenizer_path}
    external_lib: null
    trust_remote_code: False
    use_shm: False
    enable_gradient_checkpointing: True
    enable_activation_offload: False
    use_remove_padding: False
    lora_rank: {lora_rank}
    lora_alpha: 16
    target_modules: all-linear
    fsdp_config:
      model_dtype: bf16
      wrap_policy:
        min_num_params: 0
      cpu_offload: False
      offload_params: False

  actor:
    strategy: fsdp
    entropy_coeff: 0.0
    use_kl_loss: True
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
    clip_ratio: 0.2
    clip_ratio_low: 0.2
    clip_ratio_high: 0.2
    loss_agg_mode: token-mean
    ppo_epochs: 1
    ppo_mini_batch_size: {train_batch_size}
    ppo_micro_batch_size: null
    ppo_micro_batch_size_per_gpu: {ppo_micro_batch_size_per_gpu}
    ppo_max_token_len_per_gpu: 4096
    use_dynamic_bsz: False
    grad_clip: 1.0
    use_torch_compile: False
    data_loader_seed: null
    shuffle: False
    ulysses_sequence_parallel_size: 1
    optim:
      lr: {learning_rate}
      weight_decay: 0.01
      warmup_steps_ratio: 0.03
      clip_grad: 1.0
      lr_scheduler: cosine
    fsdp_config:
      wrap_policy:
        min_num_params: 0
      param_offload: False
      optimizer_offload: False
      fsdp_size: -1
    checkpoint:
      contents: ['model', 'optimizer', 'extra']
      save_contents: ['model', 'optimizer', 'extra']
      load_contents: ['model', 'optimizer', 'extra']

  ref:
    fsdp_config:
      wrap_policy:
        min_num_params: 0
      param_offload: False
      fsdp_size: -1
    log_prob_micro_batch_size: null
    log_prob_micro_batch_size_per_gpu: {log_prob_micro_batch_size_per_gpu}
    ulysses_sequence_parallel_size: 1

  rollout:
    name: vllm
    mode: sync
    load_format: safetensors
    n: {rollout_n}
    log_every_n_steps: {rollout_log_every_n_steps}
    temperature: 1.0
    top_k: -1
    top_p: 1.0
    do_sample: True
    prompt_length: 512
    response_length: 32
    max_model_len: 544
    tensor_model_parallel_size: 1
    dtype: bfloat16
    gpu_memory_utilization: {rollout_gpu_memory_utilization}
    ignore_eos: False
    enforce_eager: True
    free_cache_engine: True
    disable_log_stats: False
    enable_chunked_prefill: False
    log_prob_micro_batch_size: null
    log_prob_micro_batch_size_per_gpu: {log_prob_micro_batch_size_per_gpu}
    log_prob_max_token_len_per_gpu: 544
    log_prob_use_dynamic_bsz: False
    layered_summon: True
    calculate_log_probs: False
    engine_kwargs:
      vllm: {{}}
      sglang: {{}}
    multi_turn:
      enable: False
    val_kwargs:
      n: 1
      do_sample: False
      temperature: 0.0
      top_k: -1
      top_p: 1.0

critic:
  strategy: fsdp
  use_dynamic_bsz: False
  model:
    path: {model_path}
    enable_gradient_checkpointing: False
    use_remove_padding: False
    fsdp_config:
      model_dtype: bf16
      wrap_policy:
        min_num_params: 0
      cpu_offload: False
      offload_params: False
  optim:
    lr: 1e-5
  ppo_mini_batch_size: {train_batch_size}
  ppo_micro_batch_size: null
  ppo_micro_batch_size_per_gpu: {ppo_micro_batch_size_per_gpu}

trainer:
  device: cuda
  balance_batch: False
  npu_profile:
    options: {{}}
  total_epochs: 1
  total_training_steps: {total_training_steps}
  project_name: fact_check_grpo
  experiment_name: {experiment_name}
  logger: ['console', 'tensorboard']
  log_val_generations: 0
  nnodes: 1
  n_gpus_per_node: 1
  save_freq: {validate_every}
  test_freq: {validate_every}
  val_before_train: False
  critic_warmup: 0
  default_hdfs_dir: null
  default_local_dir: {checkpoint_dir}
  validation_data_dir: {validation_dir}
  rollout_data_dir: {rollout_dir}
  resume_mode: auto
  resume_from_path: null
  remove_previous_ckpt_in_save: False
  del_local_ckpt_after_load: False
  max_actor_ckpt_to_keep: 2
  max_critic_ckpt_to_keep: 2
  ray_wait_register_center_timeout: 300
"""


def render_prepare_command(
    model_path: str,
    train_ratio: float,
    drop_nei: bool,
    ray_num_cpus: int,
    rollout_gpu_memory_utilization: float,
    rollout_log_every_n_steps: int,
    smoke_train_examples: int,
    smoke_valid_examples: int,
    formal_valid_examples: int,
    formal_valid_seed: int,
) -> str:
    cmd = [
        "python",
        "train_grpo.py",
        f"--model-path {model_path}",
        f"--tokenizer-path {model_path}",
        f"--train-ratio {train_ratio}",
        f"--ray-num-cpus {ray_num_cpus}",
        f"--rollout-gpu-memory-utilization {rollout_gpu_memory_utilization}",
        f"--rollout-log-every-n-steps {rollout_log_every_n_steps}",
        f"--smoke-train-examples {smoke_train_examples}",
        f"--smoke-valid-examples {smoke_valid_examples}",
        f"--formal-valid-examples {formal_valid_examples}",
        f"--formal-valid-seed {formal_valid_seed}",
    ]
    if drop_nei:
        cmd.append("--drop-nei")
    return " \\\n  ".join(cmd)


def render_runner_command(config_path: Path, stage: str, max_steps: int, chunk_size: int) -> str:
    return (
        "python run_grpo_experiment.py "
        f"--config {config_path} --stage {stage} --chunk-size {chunk_size} --max-steps {max_steps}"
    )


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--drop-nei", action="store_true")

    parser.add_argument("--smoke-train-batch-size", type=int, default=8)
    parser.add_argument("--formal-train-batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=8)

    parser.add_argument("--rollout-n", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--ppo-micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--log-prob-micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--smoke-lr", type=float, default=1e-5)
    parser.add_argument("--formal-lr", type=float, default=8e-6)
    parser.add_argument("--ray-num-cpus", type=int, default=DEFAULT_RAY_NUM_CPUS)
    parser.add_argument(
        "--rollout-gpu-memory-utilization",
        type=float,
        default=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument(
        "--rollout-log-every-n-steps",
        type=int,
        default=DEFAULT_ROLLOUT_LOG_EVERY_N_STEPS,
    )

    parser.add_argument("--smoke-train-examples", type=int, default=32)
    parser.add_argument("--smoke-valid-examples", type=int, default=10)
    parser.add_argument("--formal-valid-examples", type=int, default=DEFAULT_FORMAL_VALID_EXAMPLES)
    parser.add_argument("--formal-valid-seed", type=int, default=DEFAULT_FORMAL_VALID_SEED)

    args = parser.parse_args()

    if not (0.05 <= args.rollout_gpu_memory_utilization <= 0.95):
        raise ValueError("--rollout-gpu-memory-utilization must be between 0.05 and 0.95")
    if args.rollout_log_every_n_steps <= 0:
        raise ValueError("--rollout-log-every-n-steps must be positive")
    if args.smoke_train_examples <= 0:
        raise ValueError("--smoke-train-examples must be positive")
    if args.smoke_valid_examples <= 0:
        raise ValueError("--smoke-valid-examples must be positive")
    if args.formal_valid_examples <= 0:
        raise ValueError("--formal-valid-examples must be positive")

    model_dir = _resolve_existing_dir(args.model_path, "--model-path")
    tokenizer_dir = _resolve_existing_dir(
        args.tokenizer_path if args.tokenizer_path else args.model_path,
        "--tokenizer-path",
    )

    path_cfg = PathConfig()
    ensure_dirs(path_cfg)

    if not ShardedBM25SentenceEnv.exists(path_cfg):
        raise FileNotFoundError(
            f"Missing BM25 shards under {path_cfg.bm25_shard_dir}. Run build_corpus.py first."
        )

    dataset_summary = build_and_export_default(
        path_cfg=path_cfg,
        run_cfg=RunConfig(train_ratio=args.train_ratio),
        train_ratio=args.train_ratio,
        drop_nei=args.drop_nei,
    )

    smoke_train_path = path_cfg.export_dir / "train_smoke.parquet"
    smoke_valid_path = path_cfg.export_dir / "valid_smoke.parquet"
    smoke_summary = build_smoke_parquet(
        full_train_path=path_cfg.train_for_verl_path,
        full_valid_path=path_cfg.valid_for_verl_path,
        smoke_train_path=smoke_train_path,
        smoke_valid_path=smoke_valid_path,
        smoke_train_examples=args.smoke_train_examples,
        smoke_valid_examples=args.smoke_valid_examples,
    )

    formal_valid_path = path_cfg.export_dir / "valid_formal_subset.parquet"
    formal_valid_summary = build_subset_parquet(
        full_path=path_cfg.valid_for_verl_path,
        subset_path=formal_valid_path,
        subset_examples=args.formal_valid_examples,
        seed=args.formal_valid_seed,
        shuffle=True,
    )

    smoke_name = "bm25_query_grpo_smoke"
    formal_name = "bm25_query_grpo_formal"

    smoke_yaml = render_yaml(
        train_file=str(smoke_train_path),
        val_file=str(smoke_valid_path),
        path_cfg=path_cfg,
        model_path=str(model_dir),
        tokenizer_path=str(tokenizer_dir),
        experiment_name=smoke_name,
        train_batch_size=args.smoke_train_batch_size,
        val_batch_size=args.val_batch_size,
        rollout_n=args.rollout_n,
        ppo_micro_batch_size_per_gpu=args.ppo_micro_batch_size_per_gpu,
        log_prob_micro_batch_size_per_gpu=args.log_prob_micro_batch_size_per_gpu,
        total_training_steps=SMOKE_STEPS,
        validate_every=SMOKE_VALIDATE_EVERY,
        learning_rate=args.smoke_lr,
        lora_rank=args.lora_rank,
        ray_num_cpus=args.ray_num_cpus,
        rollout_gpu_memory_utilization=args.rollout_gpu_memory_utilization,
        rollout_log_every_n_steps=args.rollout_log_every_n_steps,
    )

    formal_yaml = render_yaml(
        train_file=str(path_cfg.train_for_verl_path),
        val_file=str(formal_valid_path),
        path_cfg=path_cfg,
        model_path=str(model_dir),
        tokenizer_path=str(tokenizer_dir),
        experiment_name=formal_name,
        train_batch_size=args.formal_train_batch_size,
        val_batch_size=args.val_batch_size,
        rollout_n=args.rollout_n,
        ppo_micro_batch_size_per_gpu=args.ppo_micro_batch_size_per_gpu,
        log_prob_micro_batch_size_per_gpu=args.log_prob_micro_batch_size_per_gpu,
        total_training_steps=FORMAL_STEPS,
        validate_every=FORMAL_VALIDATE_EVERY,
        learning_rate=args.formal_lr,
        lora_rank=args.lora_rank,
        ray_num_cpus=args.ray_num_cpus,
        rollout_gpu_memory_utilization=args.rollout_gpu_memory_utilization,
        rollout_log_every_n_steps=args.rollout_log_every_n_steps,
    )

    smoke_cfg_path = path_cfg.experiment_configs_dir / f"{smoke_name}.yaml"
    formal_cfg_path = path_cfg.experiment_configs_dir / f"{formal_name}.yaml"
    write_text(smoke_cfg_path, smoke_yaml)
    write_text(formal_cfg_path, formal_yaml)

    path_cfg.verl_config_path.write_text(formal_yaml, encoding="utf-8")

    smoke_script = path_cfg.experiment_scripts_dir / "run_smoke_grpo.sh"
    formal_script = path_cfg.experiment_scripts_dir / "run_formal_grpo.sh"
    baseline_script = path_cfg.experiment_scripts_dir / "run_validation_baseline.sh"

    write_text(
        smoke_script,
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + render_runner_command(
            smoke_cfg_path,
            "smoke",
            SMOKE_STEPS,
            SMOKE_VALIDATE_EVERY,
        )
        + "\n",
        executable=True,
    )

    write_text(
        formal_script,
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + render_runner_command(
            formal_cfg_path,
            "formal",
            FORMAL_STEPS,
            FORMAL_VALIDATE_EVERY,
        )
        + "\n",
        executable=True,
    )

    write_text(
        baseline_script,
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "python eval_baseline.py "
        + f"--input-path {path_cfg.valid_for_verl_path} --strategy entity --save-details "
        + f"--details-path {path_cfg.experiment_tables_dir / formal_name / 'baseline_entity_details.jsonl'} --max-examples 1000000\n",
        executable=True,
    )

    path_cfg.verl_command_path.write_text(
        render_runner_command(
            formal_cfg_path,
            "formal",
            FORMAL_STEPS,
            FORMAL_VALIDATE_EVERY,
        ) + "\n",
        encoding="utf-8",
    )
    path_cfg.verl_command_path.chmod(0o755)

    payload = {
        "dataset": dataset_summary,
        "smoke_dataset": smoke_summary,
        "formal_validation_subset": formal_valid_summary,
        "formal_full_validation_path": str(path_cfg.valid_for_verl_path),
        "verl_installed": importlib.util.find_spec("verl") is not None,
        "model_path": str(model_dir),
        "tokenizer_path": str(tokenizer_dir),
        "prepare_command": render_prepare_command(
            model_path=str(model_dir),
            train_ratio=args.train_ratio,
            drop_nei=args.drop_nei,
            ray_num_cpus=args.ray_num_cpus,
            rollout_gpu_memory_utilization=args.rollout_gpu_memory_utilization,
            rollout_log_every_n_steps=args.rollout_log_every_n_steps,
            smoke_train_examples=args.smoke_train_examples,
            smoke_valid_examples=args.smoke_valid_examples,
            formal_valid_examples=args.formal_valid_examples,
            formal_valid_seed=args.formal_valid_seed,
        ),
        "configs": {
            "smoke": str(smoke_cfg_path),
            "formal": str(formal_cfg_path),
        },
        "scripts": {
            "smoke": str(smoke_script),
            "formal": str(formal_script),
            "baseline": str(baseline_script),
        },
        "training_rules": {
            "smoke": {
                "max_steps": SMOKE_STEPS,
                "validate_every": SMOKE_VALIDATE_EVERY,
                "train_examples": smoke_summary["smoke_train_examples"],
                "valid_examples": smoke_summary["smoke_valid_examples"],
            },
            "formal": {
                "max_steps": FORMAL_STEPS,
                "validate_every": FORMAL_VALIDATE_EVERY,
                "monitor_valid_examples": formal_valid_summary["examples"],
                "full_valid_examples": dataset_summary["valid_examples"],
                "early_stopping_patience": 4,
                "min_recall10_improve": 0.005,
            },
        },
        "rollout_gpu_memory_utilization": args.rollout_gpu_memory_utilization,
        "rollout_log_every_n_steps": args.rollout_log_every_n_steps,
        "outputs_root": str(path_cfg.experiment_root),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
