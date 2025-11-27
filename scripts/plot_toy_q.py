import argparse
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

TOY_ENV_KEY = "ToyPERBias-v0"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_q_values_for_agent(agent: str, results_root: str = "results") -> Tuple[List[int], List[np.ndarray], str]:
    """
    Load q_values.npy for all seeds of a given agent.

    Returns:
        seeds: list of seed integers
        q_tables: list of numpy arrays (one per seed), shape [n_states, n_actions]
        agent_dir: path to results/<agent> directory
    """
    agent_dir = os.path.join(results_root, TOY_ENV_KEY, agent)
    if not os.path.isdir(agent_dir):
        raise FileNotFoundError(f"Agent directory not found: {agent_dir}")

    seeds = []
    q_tables = []

    for entry in sorted(os.listdir(agent_dir)):
        if not entry.startswith("seed_"):
            continue
        seed_dir = os.path.join(agent_dir, entry)
        q_path = os.path.join(seed_dir, "q_values.npy")
        if not os.path.isfile(q_path):
            print(f"[WARN] No q_values.npy in {seed_dir}, skipping.")
            continue
        try:
            seed_str = entry.split("_")[-1]
            seed = int(seed_str)
        except ValueError:
            print(f"[WARN] Could not parse seed from folder name {entry}, skipping.")
            continue

        q = np.load(q_path)
        q_tables.append(q)
        seeds.append(seed)
        print(f"[INFO] Loaded Q-table for agent={agent}, seed={seed} from {q_path}")

    if not q_tables:
        raise RuntimeError(f"No q_values.npy files found for agent '{agent}' in {agent_dir}")

    return seeds, q_tables, agent_dir


def compute_values(q_values: np.ndarray) -> np.ndarray:
    """
    Given Q-table [n_states, n_actions], compute:
      V(s) = max_a Q(s,a)
    Returns:
      values: shape [n_states]
    """
    return q_values.max(axis=1)


def make_state_names(n_states: int) -> List[str]:
    if n_states < 4:
        # Fallback: just s0, s1, ...
        return [f"s{i}" for i in range(n_states)]

    # 1 (S0) + K (safe) + 3 (R,G,H) = n_states  =>  K = n_states - 4
    safe_chain_len = n_states - 4
    names = ["S0(start)"]
    names += [f"S{i}" for i in range(1, safe_chain_len + 1)]
    names += ["R(risky)", "G(goal)", "H(hole)"]
    return names


ACTION_NAMES = [
    "a0 (safe/cont)",
    "a1 (risky)",
]


def plot_v_bar(values: np.ndarray, out_path: str, title: str):
    """
    Bar plot of V(s) over states for the toy env.
    """
    n_states = values.shape[0]
    x = np.arange(n_states)
    state_names = make_state_names(n_states)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, values, align="center")

    ax.set_xticks(x)
    ax.set_xticklabels(state_names, rotation=45, ha="right")

    ax.set_ylabel("V(s) = max_a Q(s,a)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved V(s) bar plot to {out_path}")


def plot_q_bar(q_values: np.ndarray, out_path: str, title: str):
    """
    Grouped bar plot of Q(s,a) for each state and action.
    For the toy env:
      - action 0 ~ safe / continue
      - action 1 ~ risky (only meaningful at S0/R)
    """
    n_states, n_actions = q_values.shape
    x = np.arange(n_states)
    width = 0.35 if n_actions == 2 else 0.8 / max(n_actions, 1)
    state_names = make_state_names(n_states)

    fig, ax = plt.subplots(figsize=(8, 4))

    for a in range(n_actions):
        offset = (a - (n_actions - 1) / 2.0) * width
        label = ACTION_NAMES[a] if a < len(ACTION_NAMES) else f"a{a}"
        ax.bar(x + offset, q_values[:, a], width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(state_names, rotation=45, ha="right")

    ax.set_ylabel("Q(s,a)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved Q(s,a) bar plot to {out_path}")


def make_all_plots_for_q(q_values: np.ndarray, out_dir: str, title_prefix: str):
    """
    Generate V(s) and Q(s,a) plots for the toy PER-bias environment.
    """
    ensure_dir(out_dir)

    values = compute_values(q_values)

    # V(s)
    plot_v_bar(
        values,
        out_path=os.path.join(out_dir, "v_bar.png"),
        title=f"{title_prefix} - State values V(s)",
    )

    # Q(s,a)
    plot_q_bar(
        q_values,
        out_path=os.path.join(out_dir, "q_bar.png"),
        title=f"{title_prefix} - Action values Q(s,a)",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Q and V for the ToyPERBias environment. "
            "Requires q_values.npy under results/ToyPERBias-v0/<agent>/seed_XX/."
        )
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Agent key (e.g. per, per_sib_sample, per_sib_avg).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory where run_experiments.py saved results.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="If set, also compute and plot Q/V averaged across seeds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    seeds, q_tables, agent_dir = load_q_values_for_agent(
        agent=args.agent,
        results_root=args.results_root,
    )

    # Per-seed plots
    for seed, q_values in zip(seeds, q_tables):
        seed_dir = os.path.join(agent_dir, f"seed_{seed:02d}")
        out_dir = os.path.join(seed_dir, "plots_toy_qv")
        title_prefix = f"Agent={args.agent}, seed={seed}"
        print(f"[INFO] Making V/Q plots for {title_prefix}")
        make_all_plots_for_q(q_values, out_dir, title_prefix)

    # Aggregate (mean over seeds)
    if args.aggregate:
        print("[INFO] Computing aggregated Q-table (mean over seeds).")
        q_stack = np.stack(q_tables, axis=0)  # [n_seeds, n_states, n_actions]
        q_mean = q_stack.mean(axis=0)
        agg_dir = os.path.join(agent_dir, "plots_toy_qv_aggregate")
        title_prefix = f"Agent={args.agent}, aggregated over {len(seeds)} seeds"
        make_all_plots_for_q(q_mean, agg_dir, title_prefix)


if __name__ == "__main__":
    main()
