# scripts/plot_q_tables.py
import argparse
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_q_values_for_agent(agent: str, results_root: str = "results") -> Tuple[List[int], List[np.ndarray], str]:
    """
    Load q_values.npy for all seeds of a given agent.

    Returns:
        seeds: list of seed integers
        q_tables: list of numpy arrays (one per seed), shape [n_states, n_actions]
        agent_dir: path to results/<agent> directory
    """
    agent_dir = os.path.join(results_root, agent)
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_grid_size_from_q(q_values: np.ndarray) -> int:
    """
    Given Q-table of shape [n_states, n_actions], infer n x n grid.
    """
    n_states = q_values.shape[0]
    n = int(np.sqrt(n_states))
    if n * n != n_states:
        raise ValueError(f"State count {n_states} is not a perfect square; cannot map to a grid.")
    return n


def compute_values_and_policy(q_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given Q-table [n_states, n_actions], compute:
      V(s) = max_a Q(s,a)
      pi(s) = argmax_a Q(s,a)
    Returns:
      values: shape [n_states]
      policy: shape [n_states] (action indices)
    """
    values = q_values.max(axis=1)
    policy = q_values.argmax(axis=1)
    return values, policy


def next_state_deterministic(s: int, a: int, n: int) -> int:
    """
    Deterministic next state on an n x n grid for actions:
      0 = left, 1 = down, 2 = right, 3 = up
    Ignores holes/slip, just moves within bounds.
    """
    r, c = divmod(s, n)
    if a == 0:       # left
        c = max(0, c - 1)
    elif a == 1:     # down
        r = min(n - 1, r + 1)
    elif a == 2:     # right
        c = min(n - 1, c + 1)
    elif a == 3:     # up
        r = max(0, r - 1)
    return r * n + c


def compute_greedy_path(policy: np.ndarray, n: int, max_steps: int = 100) -> List[int]:
    """
    Compute a greedy path from state 0 to state n*n-1 under the given policy.
    Stops if:
      - goal is reached
      - a loop is detected
      - max_steps is exceeded
    """
    start = 0
    goal = n * n - 1
    s = start
    path = [s]
    visited = set()

    for _ in range(max_steps):
        if s == goal:
            break
        if s in visited:
            print("[INFO] Greedy path entered a loop, stopping.")
            break
        visited.add(s)
        a = int(policy[s])
        s_next = next_state_deterministic(s, a, n)
        if s_next == s:
            # stuck at boundary
            print("[INFO] Greedy path stuck at boundary, stopping.")
            break
        path.append(s_next)
        s = s_next

    return path


def plot_value_heatmap(values: np.ndarray, out_path: str, title: str):
    """
    Plot V(s) = max_a Q(s,a) as an n x n heatmap with numeric annotations.
    """
    n = get_grid_size_from_q(values.reshape(-1, 1))
    grid = values.reshape(n, n)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap="viridis")

    for i in range(n):
        for j in range(n):
            v = grid[i, j]
            text_color = "white" if v > grid.mean() else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=text_color, fontsize=7)

    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved value heatmap to {out_path}")


def plot_q_heatmaps(q_values: np.ndarray, out_dir: str, title_prefix: str):
    """
    Plot per-action Q(s,a) heatmaps, one figure per action.
    """
    ensure_dir(out_dir)
    n_states, n_actions = q_values.shape
    n = get_grid_size_from_q(q_values)

    action_names = ["left (0)", "down (1)", "right (2)", "up (3)"]
    for a in range(n_actions):
        grid = q_values[:, a].reshape(n, n)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(grid, cmap="coolwarm")

        for i in range(n):
            for j in range(n):
                v = grid[i, j]
                text_color = "white" if abs(v) > np.mean(np.abs(grid)) else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=text_color, fontsize=7)

        ax.set_title(f"{title_prefix} - Q(s,a) for action {action_names[a]}")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"Q(s, a={a})")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"q_heatmap_action{a}.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[INFO] Saved action-Q heatmap to {out_path}")


def plot_policy_arrows(q_values: np.ndarray, out_path: str, title: str):
    """
    Plot greedy policy as arrows on top of V(s) heatmap.
    """
    n_states, n_actions = q_values.shape
    n = get_grid_size_from_q(q_values)

    values, policy = compute_values_and_policy(q_values)
    grid = values.reshape(n, n)

    action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap="viridis")

    for s in range(n_states):
        r, c = divmod(s, n)
        a = int(policy[s])
        symbol = action_symbols.get(a, "?")
        text_color = "white" if grid[r, c] > grid.mean() else "black"
        ax.text(c, r, symbol, ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved policy arrows plot to {out_path}")


def plot_policy_with_path(q_values: np.ndarray, out_path: str, title: str):
    """
    Plot V(s) heatmap with greedy policy arrows and highlight the greedy path
    from state 0 to state n*n-1.
    """
    n_states, n_actions = q_values.shape
    n = get_grid_size_from_q(q_values)

    values, policy = compute_values_and_policy(q_values)
    grid = values.reshape(n, n)

    path = compute_greedy_path(policy, n, max_steps=5 * n * n)
    path_set = set(path)

    action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap="viridis")

    for s in range(n_states):
        r, c = divmod(s, n)
        a = int(policy[s])
        symbol = action_symbols.get(a, "?")
        text_color = "white" if grid[r, c] > grid.mean() else "black"

        # highlight path cells with a different text color or bounding box
        if s in path_set:
            # a simple red outline rectangle around the cell
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0,
                                 edgecolor="red", facecolor="none", linewidth=2)
            ax.add_patch(rect)

        ax.text(c, r, symbol, ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

    # mark start and goal explicitly
    start = 0
    goal = n * n - 1
    sr, sc = divmod(start, n)
    gr, gc = divmod(goal, n)
    ax.text(sc, sr, "S", ha="center", va="center", color="cyan", fontsize=12, fontweight="bold")
    ax.text(gc, gr, "G", ha="center", va="center", color="magenta", fontsize=12, fontweight="bold")

    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved policy+path plot to {out_path}")


def make_all_plots_for_q(
    q_values: np.ndarray,
    out_dir: str,
    title_prefix: str,
):
    """
    Generate all plots for a single Q-table:
      - value heatmap
      - per-action Q heatmaps
      - policy arrows
      - policy with greedy path
    """
    ensure_dir(out_dir)

    values, _ = compute_values_and_policy(q_values)

    # 1. Value heatmap
    plot_value_heatmap(
        values,
        out_path=os.path.join(out_dir, "values_heatmap.png"),
        title=f"{title_prefix} - State values V(s)",
    )

    # 2. Per-action Q heatmaps
    plot_q_heatmaps(
        q_values,
        out_dir=os.path.join(out_dir, "q_heatmaps"),
        title_prefix=title_prefix,
    )

    # 3. Policy arrows
    plot_policy_arrows(
        q_values,
        out_path=os.path.join(out_dir, "policy_arrows.png"),
        title=f"{title_prefix} - Greedy policy (arrows)",
    )

    # 4. Policy + greedy path
    plot_policy_with_path(
        q_values,
        out_path=os.path.join(out_dir, "policy_with_path.png"),
        title=f"{title_prefix} - Greedy path from start to goal",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Q-tables for FrozenLake-style grids. "
            "Requires q_values.npy under results/<agent>/seed_XX/."
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
        help="If set, also compute and plot Q-values averaged across seeds.",
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
        out_dir = os.path.join(seed_dir, "plots_q")
        title_prefix = f"Agent={args.agent}, seed={seed}"
        print(f"[INFO] Making plots for {title_prefix}")
        make_all_plots_for_q(q_values, out_dir, title_prefix)

    # Aggregate (mean over seeds)
    if args.aggregate:
        print("[INFO] Computing aggregated Q-table (mean over seeds).")
        q_stack = np.stack(q_tables, axis=0)  # [n_seeds, n_states, n_actions]
        q_mean = q_stack.mean(axis=0)
        agg_dir = os.path.join(agent_dir, "plots_q_aggregate")
        title_prefix = f"Agent={args.agent}, aggregated over {len(seeds)} seeds"
        make_all_plots_for_q(q_mean, agg_dir, title_prefix)


if __name__ == "__main__":
    main()
