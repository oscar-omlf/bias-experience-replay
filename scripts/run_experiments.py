import argparse
import json
import os
from typing import Dict, List, Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose

from scripts.train import run_training
from src.utils.seed import set_global_seeds


AGENT_VARIANTS = {
    "dqn": {
        "label": "DQN",
        "group": "DQN",
        "overrides": [
            "agents=dqn",
        ],
    },
    "per": {
        "label": "PER",
        "group": "PER_NOMIT",
        "overrides": [
            "agents=per",
            "agents.replay.sa_mitigation.enabled=false",
            "agents.replay.sa_mitigation.method=none",
            "agents.replay.sa_mitigation.max_group=0",
        ],
    },
    "per_sib_sample": {
        "label": "PER+SiblingSampling",
        "group": "PER_SAMPLE",
        "overrides": [
            "agents=per",
            "agents.replay.sa_mitigation.enabled=true",
            "agents.replay.sa_mitigation.method=sample",
            "agents.replay.sa_mitigation.update_all_siblings=false",
            "agents.replay.sa_mitigation.max_group=0",
        ],
    },
    "per_sib_avg": {
        "label": "PER+SiblingAveraging",
        "group": "PER_AVG",
        "overrides": [
            "agents=per",
            "agents.replay.sa_mitigation.enabled=true",
            "agents.replay.sa_mitigation.method=avg",
            "agents.replay.sa_mitigation.update_all_siblings=true",
        ],
    },
    "per_model": {
        "label": "PER+Model",
        "group": "PER_MODEL",
        "overrides": [
            "agents=per",
            "agents.replay.sa_mitigation.enabled=true",
            "agents.replay.sa_mitigation.method=model",
            "agents.replay.sa_mitigation.update_all_siblings=false",
            "agents.replay.sa_mitigation.max_group=0",
        ],
    },
}



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_dicts_to_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    # Collect union of keys
    fieldnames = sorted({k for row in rows for k in row.keys()})
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_cfg_for_run(
    base_overrides: List[str],
    agent_overrides: List[str],
    seed: int,
) -> DictConfig:
    """
    Uses Hydra programmatic API to compose a config with the given overrides.
    """
    overrides = list(base_overrides) + list(agent_overrides) + [f"seed={seed}"]

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def run_single_seed(
    agent_key: str,
    seed: int,
    base_overrides: List[str],
    results_root: str = "results",
):
    """
    Runs one full training for a given agent variant and seed.

    Saves:
      - config.yaml
      - metrics CSVs
      - numpy arrays for main metrics
      - summary.json
    """
    variant = AGENT_VARIANTS[agent_key]
    agent_label = variant["label"]

    set_global_seeds(seed)

    cfg = build_cfg_for_run(
        base_overrides=base_overrides,
        agent_overrides=variant["overrides"],
        seed=seed,
    )

    new_agent_key = agent_key
    if agent_key == "per_sib_avg":
        max_group = cfg.agents.replay.sa_mitigation.max_group
        new_agent_key = f"{agent_key}_g{max_group}"

    env_id = cfg.env.id
    map_name = getattr(cfg.env, "map_name", None)
    if map_name:
        env_key = f"{env_id}-{map_name}"
    else:
        env_key = env_id

    # W&B settings for this experiment
    cfg.wandb.job_type = "training"

    base_group = variant.get("group", agent_key)
    if agent_key == "per_sib_avg":
        max_group = cfg.agents.replay.sa_mitigation.max_group
        cfg.wandb.group = f"{base_group}_g{max_group}"
    else:
        cfg.wandb.group = base_group

    summary = run_training(cfg)

    # Build result directory: results/<env_key>/<agent_key>/seed_XX/
    agent_dir = os.path.join(results_root, env_key, new_agent_key)
    seed_dir = os.path.join(agent_dir, f"seed_{seed:02d}")
    ensure_dir(seed_dir)

    # Save config used
    OmegaConf.save(cfg, os.path.join(seed_dir, "config.yaml"))

    # Save logs as CSV
    episode_logs = summary.get("episode_logs", [])
    step_logs = summary.get("step_logs", [])
    eval_logs = summary.get("eval_logs", [])

    save_dicts_to_csv(os.path.join(seed_dir, "episode_logs.csv"), episode_logs)
    save_dicts_to_csv(os.path.join(seed_dir, "step_logs.csv"), step_logs)
    save_dicts_to_csv(os.path.join(seed_dir, "eval_logs.csv"), eval_logs)

    # Numpy arrays for convenience
    if episode_logs:
        ep_returns = np.array(
            [row.get("train/episode_return", 0.0) for row in episode_logs],
            dtype=np.float32,
        )
        np.save(os.path.join(seed_dir, "episode_returns.npy"), ep_returns)

        ep_success = np.array(
            [row.get("train/episode_success", 0.0) for row in episode_logs],
            dtype=np.float32,
        )
        np.save(os.path.join(seed_dir, "episode_success.npy"), ep_success)

    if eval_logs:
        eval_success = np.array(
            [row.get("eval/success_rate", 0.0) for row in eval_logs],
            dtype=np.float32,
        )
        np.save(os.path.join(seed_dir, "eval_success_rates.npy"), eval_success)

        eval_return_mean = np.array(
            [row.get("eval/return_mean", 0.0) for row in eval_logs],
            dtype=np.float32,
        )
        np.save(os.path.join(seed_dir, "eval_return_mean.npy"), eval_return_mean)

    q_values = summary.get("q_values", None)
    if q_values is not None:
        np.save(os.path.join(seed_dir, "q_values.npy"), q_values)

    # Per-seed summary
    final_eval = summary.get("final_eval", {})
    seed_summary = {
        "agent_key": new_agent_key,
        "agent_label": agent_label,
        "env_key": env_key,
        "seed": seed,
        "total_steps": int(summary.get("total_steps", 0)),
        "final_eval": final_eval,
        "num_episodes": len(episode_logs),
        "num_evals": len(eval_logs),
    }
    save_json(os.path.join(seed_dir, "summary.json"), seed_summary)

    return seed_summary


def run_multi_seed(
    agent_key: str,
    seeds: List[int],
    base_overrides: List[str],
    results_root: str = "results",
):
    """
    Runs multiple seeds for a single agent variant and writes
    a cross-seed summary JSON.
    """
    all_seed_summaries = []
    for seed in seeds:
        print(f"\n=== Running {agent_key} seed={seed} ===")
        s_sum = run_single_seed(
            agent_key=agent_key,
            seed=seed,
            base_overrides=base_overrides,
            results_root=results_root,
        )
        all_seed_summaries.append(s_sum)

    if not all_seed_summaries:
        return

    # All runs should share the same env. take from the first
    env_key = all_seed_summaries[0].get("env_key", "env")

    # Cross-seed summary
    success_list = []
    return_list = []
    for s in all_seed_summaries:
        fe = s.get("final_eval", {})
        if "eval/success_rate" in fe:
            success_list.append(float(fe["eval/success_rate"]))
        if "eval/return_mean" in fe:
            return_list.append(float(fe["eval/return_mean"]))

    agent_key_for_dir = all_seed_summaries[0].get("agent_key", agent_key)
    agent_dir = os.path.join(results_root, env_key, agent_key_for_dir)
    ensure_dir(agent_dir)

    cross_summary = {
        "env_key": env_key,
        "agent_key": agent_key_for_dir,
        "agent_label": AGENT_VARIANTS[agent_key]["label"],
        "seeds": [int(s["seed"]) for s in all_seed_summaries],
        "n_seeds": len(all_seed_summaries),
    }

    if success_list:
        success_arr = np.array(success_list, dtype=np.float32)
        cross_summary.update(
            {
                "final_success_mean": float(success_arr.mean()),
                "final_success_std": float(success_arr.std()),
            }
        )
    if return_list:
        return_arr = np.array(return_list, dtype=np.float32)
        cross_summary.update(
            {
                "final_return_mean": float(return_arr.mean()),
                "final_return_std": float(return_arr.std()),
            }
        )

    save_json(os.path.join(agent_dir, "summary.json"), cross_summary)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-seed PER bias experiments on FrozenLake-8x8.")
    parser.add_argument(
        "--agent",
        type=str,
        choices=list(AGENT_VARIANTS.keys()),
        default="per",
        help="Which agent variant to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="List of integer seeds, e.g. --seeds 0 1 2 3 4",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory for saving results.",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Additional Hydra-style overrides, e.g. env=frozenlake8x8 train.total_steps=200000",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_overrides = list(args.override)

    run_multi_seed(
        agent_key=args.agent,
        seeds=args.seeds,
        base_overrides=base_overrides,
        results_root=args.results_root,
    )


if __name__ == "__main__":
    main()
