from __future__ import annotations

import argparse
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


X_CANDIDATES = [
    "global_step",
    "env_step",
    "env_steps",
    "timestep",
    "timesteps",
    "step",
    "steps",
    "total_steps",
    "train/total_steps",
]

def pick_x_column(df: pd.DataFrame) -> Optional[str]:
    for c in X_CANDIDATES:
        if c in df.columns:
            return c
    if "Step" in df.columns:
        return "Step"
    return None

def standard_error(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size <= 1:
        return float("nan")
    return float(np.nanstd(x, ddof=1) / math.sqrt(x.size))

def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size < 2:
        return float("nan")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_unique, idx = np.unique(x, return_index=True)
    x = x_unique
    y = y[idx]
    if x.size < 2:
        return float("nan")

    return float(np.trapezoid(y, x))

def steps_to_threshold(x: np.ndarray, y: np.ndarray, thr: float) -> float:
    """Return first x where y >= thr (linear interpolation). If never reaches, NaN."""
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size < 2:
        return float("nan")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if np.nanmax(y) < thr:
        return float("nan")

    for i in range(1, len(x)):
        if y[i] >= thr and y[i - 1] < thr:
            if y[i] == y[i - 1]:
                return float(x[i])
            t = (thr - y[i - 1]) / (y[i] - y[i - 1])
            return float(x[i - 1] + t * (x[i] - x[i - 1]))
        if y[i] >= thr and y[i - 1] >= thr:
            return float(x[i - 1])

    return float(x[np.argmax(y >= thr)])

@dataclass
class SeedSeries:
    seed: str
    x: np.ndarray
    y: np.ndarray

def read_seed_series(seed_dir: str, metric: str, prefer: str = "eval") -> Optional[SeedSeries]:
    """
    prefer:
      - "eval": read eval_logs.csv first, fallback to episode_logs.csv
      - "train": read episode_logs.csv first, fallback to eval_logs.csv
    """
    order = ["eval_logs.csv", "episode_logs.csv"] if prefer == "eval" else ["episode_logs.csv", "eval_logs.csv"]

    for fname in order:
        path = os.path.join(seed_dir, fname)
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        if df.empty:
            continue

        # metric must exist; if not, skip this file
        if metric not in df.columns:
            continue

        xcol = pick_x_column(df)
        if xcol is None:
            # fallback: index as "time"
            x = np.arange(len(df), dtype=float)
        else:
            x = df[xcol].to_numpy(dtype=float)

        y = df[metric].to_numpy(dtype=float)

        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        if len(x) == 0:
            continue

        order_idx = np.argsort(x)
        x = x[order_idx]
        y = y[order_idx]
        return SeedSeries(seed=os.path.basename(seed_dir), x=x, y=y)

    return None

def summarize_tree(results_root: str, env_key: Optional[str]) -> Dict[str, List[str]]:
    """
    Return mapping env_key -> list of agent dirs.
    Also prints a concise summary.
    """
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results root not found: {results_root}")

    env_dirs = sorted([d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))])
    if env_key is not None:
        env_dirs = [d for d in env_dirs if d == env_key]
        if not env_dirs:
            raise FileNotFoundError(
                f"env_key='{env_key}' not found under {results_root}. Available: {sorted(os.listdir(results_root))}"
            )

    mapping: Dict[str, List[str]] = {}
    for e in env_dirs:
        epath = os.path.join(results_root, e)
        agent_dirs = sorted([d for d in os.listdir(epath) if os.path.isdir(os.path.join(epath, d))])
        mapping[e] = agent_dirs

    print("Results tree summary")
    for e, agents in mapping.items():
        print(f"- {e}/")
        for a in agents:
            apath = os.path.join(results_root, e, a)
            seed_dirs = sorted(
                [d for d in os.listdir(apath) if os.path.isdir(os.path.join(apath, d)) and d.startswith("seed_")]
            )
            print(f"  - {a}/  (seeds: {len(seed_dirs)})")
    return mapping

def sanitize_metric(metric: str) -> str:
    return metric.replace("/", "_").replace(" ", "_").replace(":", "_")

def bin_seed_series(x: np.ndarray, y: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    For a single seed: return y_binned where each entry is the mean y in that x-bin.
    Missing bins -> NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_binned = np.full(len(bin_edges) - 1, np.nan, dtype=float)

    idx = np.digitize(x, bin_edges) - 1
    valid = (idx >= 0) & (idx < len(y_binned)) & (~np.isnan(y))
    idx = idx[valid]
    yv = y[valid]

    if idx.size == 0:
        return y_binned

    for b in np.unique(idx):
        vals = yv[idx == b]
        if vals.size > 0:
            y_binned[b] = float(np.mean(vals))
    return y_binned

def nan_moving_average(a: np.ndarray, w: int) -> np.ndarray:
    """
    NaN-aware moving average with window w (in bins).
    """
    a = np.asarray(a, dtype=float)
    if w <= 1:
        return a.copy()

    out = np.full_like(a, np.nan, dtype=float)
    n = len(a)
    half = w // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = a[lo:hi]
        if np.all(np.isnan(window)):
            continue
        out[i] = float(np.nanmean(window))
    return out

def aggregate_binned_over_seeds(
    series_list: List[SeedSeries],
    bin_size: float,
    smooth_window: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x_centers, mean, se, n
    computed over seeds on a common x-bin grid (binning per seed, then aggregating across seeds).
    """
    if not series_list:
        return np.array([]), np.array([]), np.array([]), np.array([])

    x_min = min(float(np.nanmin(s.x)) for s in series_list)
    x_max = max(float(np.nanmax(s.x)) for s in series_list)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return np.array([]), np.array([]), np.array([]), np.array([])

    x0 = math.floor(x_min / bin_size) * bin_size
    x1 = math.ceil(x_max / bin_size) * bin_size
    if x1 <= x0:
        x1 = x0 + bin_size

    n_bins = int(round((x1 - x0) / bin_size))
    bin_edges = x0 + bin_size * np.arange(n_bins + 1, dtype=float)
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    Y = []
    for s in series_list:
        yb = bin_seed_series(s.x, s.y, bin_edges)
        Y.append(yb)
    Y = np.vstack(Y)  # (n_seeds, n_bins)

    mean = np.nanmean(Y, axis=0)
    n = np.sum(~np.isnan(Y), axis=0).astype(float)

    std = np.nanstd(Y, axis=0, ddof=1)
    se = std / np.sqrt(n)
    se[n <= 1] = np.nan

    # Smooth mean and se (optional)
    mean = nan_moving_average(mean, smooth_window)
    se = nan_moving_average(se, smooth_window)

    return x_centers, mean, se, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=str, default="results")
    p.add_argument("--env-key", type=str, default=None, help="If omitted, uses the first env folder found.")
    p.add_argument("--metric", type=str, default="eval/success_rate", help="Metric column to treat as accuracy.")
    p.add_argument(
        "--prefer",
        type=str,
        choices=["eval", "train"],
        default="eval",
        help="Which log file to prefer for the time-series (eval_logs vs episode_logs).",
    )
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.5, 0.8],
        help="Accuracy thresholds for steps-to-threshold.",
    )
    p.add_argument(
        "--bin-size",
        type=float,
        default=10000.0,
        help="Step bin width for resampling/smoothing (e.g., 5000, 10000, 20000).",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="NaN-aware moving-average window in bins (odd number recommended).",
    )
    p.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Optional plot output directory. Default: <results-root>/<env-key>/_plots",
    )
    p.add_argument(
        "--no-close",
        action="store_true",
        help="Do not close figures after saving (useful for interactive debugging).",
    )
    args = p.parse_args()

    mapping = summarize_tree(args.results_root, args.env_key)
    env_key = args.env_key or (next(iter(mapping.keys())) if mapping else None)
    if env_key is None:
        raise RuntimeError("No env folders found under results root.")

    agent_dirs = mapping[env_key]
    if not agent_dirs:
        raise RuntimeError(f"No agent folders found under {os.path.join(args.results_root, env_key)}")

    plot_dir = args.plot_dir or os.path.join(args.results_root, env_key, "_plots")
    os.makedirs(plot_dir, exist_ok=True)

    curves = {}
    seed_metrics = []

    for agent in agent_dirs:
        agent_path = os.path.join(args.results_root, env_key, agent)
        seed_dirs = sorted(
            [
                os.path.join(agent_path, d)
                for d in os.listdir(agent_path)
                if os.path.isdir(os.path.join(agent_path, d)) and d.startswith("seed_")
            ]
        )

        series_list: List[SeedSeries] = []
        aucs = []
        steps_thr = {thr: [] for thr in args.thresholds}

        for sd in seed_dirs:
            s = read_seed_series(sd, metric=args.metric, prefer=args.prefer)
            if s is None:
                continue
            series_list.append(s)

            aucs.append(auc_trapz(s.x, s.y))
            for thr in args.thresholds:
                steps_thr[thr].append(steps_to_threshold(s.x, s.y, thr))

        # Binned + smoothed curve for plotting (readability)
        x, mean, se, n = aggregate_binned_over_seeds(
            series_list,
            bin_size=args.bin_size,
            smooth_window=args.smooth_window,
        )
        if x.size == 0:
            print(f"[WARN] No usable series for agent '{agent}' (metric '{args.metric}' not found or logs missing).")
            continue

        curves[agent] = {"x": x, "mean": mean, "se": se, "n": n}

        # Sample-efficiency summary per agent
        aucs_arr = np.asarray(aucs, dtype=float)
        row = {
            "agent": agent,
            "n_seeds": len(series_list),
            "auc_mean": float(np.nanmean(aucs_arr)) if aucs_arr.size else float("nan"),
            "auc_se": standard_error(aucs_arr) if aucs_arr.size else float("nan"),
        }
        for thr in args.thresholds:
            arr = np.asarray(steps_thr[thr], dtype=float)
            row[f"steps_to_{thr:.2f}_mean"] = float(np.nanmean(arr)) if arr.size else float("nan")
            row[f"steps_to_{thr:.2f}_se"] = standard_error(arr) if arr.size else float("nan")
        seed_metrics.append(row)

    if not curves:
        raise RuntimeError("No curves were constructed. Check metric name and log CSV contents.")

    metric_tag = sanitize_metric(args.metric)

    # Plot 1: mean accuracy over time with SE, all agents (binned + smoothed)
    plt.figure()
    for agent, d in curves.items():
        x = d["x"]
        m = d["mean"]
        se = d["se"]

        plt.plot(x, m, label=agent)

        mask = np.isfinite(se) & np.isfinite(m)
        if mask.any():
            plt.fill_between(x[mask], (m - se)[mask], (m + se)[mask], alpha=0.2)

    plt.xlabel("Environment steps (binned centers)")
    plt.ylabel(args.metric)
    plt.title(
        f"{env_key}: mean {args.metric} over time (1 SE), "
        f"bin={int(args.bin_size)} steps, smooth={args.smooth_window} bins"
    )
    plt.legend()
    plt.tight_layout()

    fig1_path = os.path.join(plot_dir, f"{env_key}__{metric_tag}__mean_over_time_smoothed.png")
    plt.savefig(fig1_path, dpi=200)
    if not args.no_close:
        plt.close()

    # Plot 2: sample efficiency (AUC) with SE
    df = pd.DataFrame(seed_metrics).sort_values("agent")

    plt.figure()
    xpos = np.arange(len(df))
    plt.bar(xpos, df["auc_mean"].to_numpy(), yerr=df["auc_se"].to_numpy(), capsize=4)
    plt.xticks(xpos, df["agent"].to_list(), rotation=30, ha="right")
    plt.ylabel("AUC of accuracy vs steps")
    plt.title(f"{env_key}: sample efficiency summary (AUC, 1 SE over seeds)")
    plt.tight_layout()

    fig2_path = os.path.join(plot_dir, f"{env_key}__{metric_tag}__sample_efficiency_auc.png")
    plt.savefig(fig2_path, dpi=200)
    if not args.no_close:
        plt.close()

    print(f"\nSaved plots to:\n  {fig1_path}\n  {fig2_path}")

    print("Sample efficiency table (means, 1 SE)")
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
