import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
from collections import defaultdict

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from src.models.vqvae import VQVAE
from src.utils.seed import set_global_seeds


@dataclass
class VQVAETrainCfg:
    # environment (registry-compatible)
    env: Dict[str, Any]

    seed: int = 0
    device: str = "cpu"  # "auto" supported in main()

    # data collection
    collect_steps: int = 200_000

    # model
    codebook_size: int = 512
    embed_dim: int = 32
    hidden_channels: int = 64
    beta: float = 0.25

    # training
    batch_size: int = 256
    train_steps: int = 50_000
    lr: float = 3e-4
    recon_loss: str = "bce"  # "mse" or "bce"

    # evaluation diagnostics
    eval_sample_size: int = 50_000
    compute_sa_stats: bool = True
    sa_eval_sample_size: int = 100_000
    topk_print: int = 10
    min_count_for_sa: int = 50

    # output
    save_path: str = "artifacts/vqvae.pt"
    dump_examples_path: Optional[str] = "artifacts/vqvae_group_examples.txt"


def _parse_grid_size(v) -> Tuple[int, int]:
    """
    Accepts:
      - int (g) -> (g,g)
      - list/tuple/ListConfig [g,g] -> (g,g)
      - None -> (1,1)
    """
    if v is None:
        return (1, 1)

    if OmegaConf.is_list(v):
        if len(v) != 2:
            raise ValueError(f"vqvae.grid_size list-like value must have length 2, got {v}")
        return (int(v[0]), int(v[1]))

    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise ValueError(f"vqvae.grid_size list/tuple must have length 2, got {v}")
        return (int(v[0]), int(v[1]))

    return (int(v), int(v))


def _pack_grid_codes(codes_2d: np.ndarray, codebook_size: int) -> int:
    """
    Exact row-major base-K packing of a small code grid into a single Python int.
    """
    K = int(codebook_size)
    if K <= 1:
        raise ValueError(f"codebook_size must be >= 2 for packing; got {K}")
    flat = np.asarray(codes_2d, dtype=np.int64).ravel()
    out = 0
    base = 1
    for v in flat.tolist():
        vv = int(v)
        if vv < 0 or vv >= K:
            raise ValueError(f"grid code {vv} outside [0, {K})")
        out += vv * base
        base *= K
    return int(out)


def _effective_codebook_size(codebook_size: int, grid_size: Tuple[int, int]) -> int:
    Gh, Gw = int(grid_size[0]), int(grid_size[1])
    return int(int(codebook_size) ** int(Gh * Gw))


def _save_dataset_npz(path: str, obs_arr: np.ndarray, transitions: Optional[Dict[str, np.ndarray]] = None):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    payload: Dict[str, np.ndarray] = {"obs": np.asarray(obs_arr, dtype=np.float32)}
    if transitions is not None:
        for k, v in transitions.items():
            payload[f"trans_{k}"] = np.asarray(v)
    np.savez_compressed(path, **payload)
    print(f"[vqvae] saved dataset -> {path}")


def _load_dataset_npz(path: str) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    data = np.load(path, allow_pickle=False)
    obs_arr = np.asarray(data["obs"], dtype=np.float32)
    transitions = None
    trans_keys = [k for k in data.files if k.startswith("trans_")]
    if trans_keys:
        transitions = {k[len("trans_"):]: np.asarray(data[k]) for k in trans_keys}
    print(f"[vqvae] loaded dataset <- {path} (N={obs_arr.shape[0]})")
    return obs_arr, transitions


def _to_tensor_batch(obs_batch: np.ndarray, device: str) -> torch.Tensor:
    x = np.asarray(obs_batch, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected batch shape (B,C,H,W); got {x.shape}")
    return torch.from_numpy(x).to(device)


def _entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    c = counts.astype(np.float64)
    s = float(c.sum())
    if s <= 0:
        return 0.0
    p = c / s
    return float(-(p * np.log(p + eps)).sum())


def _ascii_mean_map(obs_batch_chw: np.ndarray) -> str:
    """
    obs_batch_chw: (N,C,H,W) binary-ish.
    Show mean occupancy over channels as ASCII.
    """
    x = np.asarray(obs_batch_chw, dtype=np.float32)
    if x.ndim != 4:
        return "<non-image obs>"

    mu = x.mean(axis=0)          # (C,H,W)
    occ = mu.sum(axis=0)         # (H,W)

    mx = float(np.max(occ)) if occ.size > 0 else 1.0
    if mx <= 0:
        mx = 1.0
    y = occ / mx

    chars = " .:-=+*#%@"
    out_lines = []
    H, W = y.shape
    for r in range(H):
        line = ""
        for c in range(W):
            idx = int(np.clip(round(y[r, c] * (len(chars) - 1)), 0, len(chars) - 1))
            line += chars[idx]
        out_lines.append(line)
    return "\n".join(out_lines)


def _compute_code_metrics(
    model: VQVAE,
    obs_arr: np.ndarray,
    device: str,
    codebook_size: int,
    eval_sample_size: int,
) -> Dict[str, float]:
    """
    Computes codebook usage, perplexity, and intra-code variance.

    Works for:
      - single-code mode: idx shape (m,)
      - grid-code mode:   idx shape (m,Gh,Gw)
    """
    xall = np.asarray(obs_arr, dtype=np.float32)
    if xall.ndim != 4:
        raise ValueError(f"Expected obs_arr shape (N,C,H,W); got {xall.shape}")

    n = xall.shape[0]
    m = min(int(eval_sample_size), n)
    idx = np.random.randint(0, n, size=m)
    sample = xall[idx]  # (m,C,H,W)

    with torch.no_grad():
        codes_t = model.encode_indices(_to_tensor_batch(sample, device=device))
        codes = codes_t.detach().cpu().numpy().astype(np.int64)

    K = int(codebook_size)

    if codes.ndim == 1:
        # (m,)
        codes_flat = codes
        Gh = Gw = 1
        any_code_mask = lambda k: (codes == k)
        uniq_per_frame = np.ones((m,), dtype=np.int64)
    elif codes.ndim == 3:
        # (m,Gh,Gw)
        Gh, Gw = int(codes.shape[1]), int(codes.shape[2])
        codes_flat = codes.reshape(-1)
        any_code_mask = lambda k: np.any(codes == k, axis=(1, 2))
        uniq_per_frame = np.asarray([len(np.unique(codes[i].ravel())) for i in range(m)], dtype=np.int64)
    else:
        raise ValueError(f"encode_indices returned unexpected shape {codes.shape}")

    counts = np.bincount(codes_flat, minlength=K).astype(np.int64)

    used = int(np.sum(counts > 0))
    usage_frac = float(used) / float(K)

    H = _entropy_from_counts(counts)
    perplexity = float(np.exp(H))

    # intra-code variance proxy on frames that contain code k (any patch)
    intra_vars = []
    max_frames_per_code = 256
    for k in range(K):
        ck = int(counts[k])
        if ck <= 0:
            continue
        mask = any_code_mask(k)
        xb = sample[mask]
        if xb.shape[0] <= 0:
            continue
        if xb.shape[0] > max_frames_per_code:
            pick = np.random.choice(xb.shape[0], size=max_frames_per_code, replace=False)
            xb = xb[pick]
        mu_k = xb.mean(axis=0)          # (C,H,W)
        var_k = float(np.mean(mu_k * (1.0 - mu_k)))
        intra_vars.append(var_k)

    intra_mean = float(np.mean(intra_vars)) if intra_vars else 0.0
    intra_p90 = float(np.quantile(np.asarray(intra_vars), 0.90)) if len(intra_vars) >= 5 else intra_mean

    out = {
        "vqvae/latent_grid_h": float(Gh),
        "vqvae/latent_grid_w": float(Gw),
        "vqvae/code_usage_frac": usage_frac,
        "vqvae/code_used": float(used),
        "vqvae/code_entropy": float(H),
        "vqvae/perplexity": float(perplexity),
        "vqvae/intra_code_var_mean": float(intra_mean),
        "vqvae/intra_code_var_p90": float(intra_p90),
        "vqvae/unique_codes_per_frame_mean": float(np.mean(uniq_per_frame)) if uniq_per_frame.size > 0 else 0.0,
        "vqvae/unique_codes_per_frame_p90": float(np.quantile(uniq_per_frame, 0.90)) if uniq_per_frame.size >= 5 else float(np.mean(uniq_per_frame)) if uniq_per_frame.size > 0 else 0.0,
    }
    return out


def _compute_sa_stats(
    model: VQVAE,
    transitions: Dict[str, np.ndarray],
    device: str,
    codebook_size: int,
    sa_eval_sample_size: int,
    min_count_for_sa: int,
    bucket_thresholds: List[int],
) -> Dict[str, float]:
    """
    Computes (state_key, action)-level diagnostics for latent grouping quality.

    - single-code mode: state_key is the code id (int)
    - grid-code mode:   state_key is a 64-bit hash of the (Gh,Gw) code grid

    Also computes entropy of next_state_key distribution within each (state_key, action).
    """
    obs = np.asarray(transitions["obs"], dtype=np.float32)
    act = np.asarray(transitions["act"], dtype=np.int64)
    rew = np.asarray(transitions["rew"], dtype=np.float32)
    done = np.asarray(transitions["done"], dtype=np.float32)
    next_obs = np.asarray(transitions["next_obs"], dtype=np.float32)

    if obs.ndim != 4 or next_obs.ndim != 4:
        raise ValueError(f"Expected obs/next_obs shape (N,C,H,W); got {obs.shape} and {next_obs.shape}")

    n = obs.shape[0]
    m = min(int(sa_eval_sample_size), n)
    idx = np.random.randint(0, n, size=m)

    obs_s = obs[idx]
    act_s = act[idx]
    rew_s = rew[idx]
    done_s = done[idx]
    next_s = next_obs[idx]

    with torch.no_grad():
        c_t = model.encode_indices(_to_tensor_batch(obs_s, device=device))
        c2_t = model.encode_indices(_to_tensor_batch(next_s, device=device))

    c = c_t.detach().cpu().numpy().astype(np.int64)
    c2 = c2_t.detach().cpu().numpy().astype(np.int64)

    # Build state keys
    if c.ndim == 1:
        keys = c.astype(np.int64)
        keys2 = c2.astype(np.int64)

        def next_entropy(counter_dict: Dict[int, int]) -> float:
            counts = np.asarray(list(counter_dict.values()), dtype=np.int64)
            return _entropy_from_counts(counts)

    # elif c.ndim == 3:
    #     keys  = np.asarray([_grid_key_from_codes(c[i])  for i in range(c.shape[0])], dtype=np.uint64)
    #     keys2 = np.asarray([_grid_key_from_codes(c2[i]) for i in range(c2.shape[0])], dtype=np.uint64)

    elif c.ndim == 3:
        keys = np.fromiter((_pack_grid_codes(c[i], codebook_size) for i in range(c.shape[0])),
                       dtype=np.int64, count=c.shape[0])
        keys2 = np.fromiter((_pack_grid_codes(c2[i], codebook_size) for i in range(c2.shape[0])),
                        dtype=np.int64, count=c2.shape[0])

        def next_entropy(counter_dict: Dict[int, int]) -> float:
            counts = np.asarray(list(counter_dict.values()), dtype=np.int64)
            return _entropy_from_counts(counts)

    else:
        raise ValueError(f"encode_indices returned unexpected shape {c.shape}")

    # Aggregate per (key, action)
    counts = defaultdict(int)
    rew_sum = defaultdict(float)
    rew_sumsq = defaultdict(float)
    done_sum = defaultdict(float)
    next_counts = defaultdict(lambda: defaultdict(int))  # per (key,a): next_key -> count

    for ki, ai, ri, di, k2i in zip(keys, act_s, rew_s, done_s, keys2):
        key = (int(ki), int(ai))
        counts[key] += 1
        rew_sum[key] += float(ri)
        rew_sumsq[key] += float(ri * ri)
        done_sum[key] += float(di)
        next_counts[key][int(k2i)] += 1

    bucket_sizes = np.asarray(list(counts.values()), dtype=np.int64)
    if bucket_sizes.size == 0:
        bucket_sizes = np.asarray([0], dtype=np.int64)

    def _q(arr, qs):
        arr = np.asarray(arr, dtype=np.float32)
        return [float(x) for x in np.quantile(arr, qs)] if arr.size > 0 else [0.0 for _ in qs]

    bs_p10, bs_p50, bs_p90 = _q(bucket_sizes, [0.10, 0.50, 0.90])
    bs_min = float(np.min(bucket_sizes))
    bs_max = float(np.max(bucket_sizes))
    bs_mean = float(np.mean(bucket_sizes))

    per_transition_bucket_size = np.asarray(
        [counts[(int(ki), int(ai))] for ki, ai in zip(keys, act_s)],
        dtype=np.int64,
    )

    frac_ge = {}
    for thr in bucket_thresholds:
        thr_i = int(thr)
        frac_ge[thr_i] = float(np.mean(per_transition_bucket_size >= thr_i))

    # summarize keys with enough support
    ent_list = []
    rstd_list = []
    done_list = []

    for key, cnt in counts.items():
        if cnt < int(min_count_for_sa):
            continue
        mu = rew_sum[key] / float(cnt)
        var = max(0.0, (rew_sumsq[key] / float(cnt)) - mu * mu)
        rstd = float(np.sqrt(var))
        dr = done_sum[key] / float(cnt)
        ent = next_entropy(next_counts[key])

        ent_list.append(ent)
        rstd_list.append(rstd)
        done_list.append(dr)

    ent_arr = np.asarray(ent_list, dtype=np.float32)
    rstd_arr = np.asarray(rstd_list, dtype=np.float32)
    done_arr = np.asarray(done_list, dtype=np.float32)

    ent_mean = float(np.mean(ent_arr)) if ent_arr.size > 0 else 0.0
    ent_p90 = float(np.quantile(ent_arr, 0.90)) if ent_arr.size >= 5 else ent_mean
    ent_max = float(np.max(ent_arr)) if ent_arr.size > 0 else 0.0

    rstd_mean = float(np.mean(rstd_arr)) if rstd_arr.size > 0 else 0.0
    rstd_p90 = float(np.quantile(rstd_arr, 0.90)) if rstd_arr.size >= 5 else rstd_mean
    rstd_max = float(np.max(rstd_arr)) if rstd_arr.size > 0 else 0.0

    done_mean = float(np.mean(done_arr)) if done_arr.size > 0 else 0.0
    done_p90 = float(np.quantile(done_arr, 0.90)) if done_arr.size >= 5 else done_mean
    done_max = float(np.max(done_arr)) if done_arr.size > 0 else 0.0

    out: Dict[str, float] = {
        "grouping/sa_pairs_counted": float(len(ent_list)),
        "grouping/sa_next_code_entropy_mean": ent_mean,
        "grouping/sa_next_code_entropy_p90": ent_p90,
        "grouping/sa_next_code_entropy_max": ent_max,
        "grouping/sa_reward_std_mean": rstd_mean,
        "grouping/sa_reward_std_p90": rstd_p90,
        "grouping/sa_reward_std_max": rstd_max,
        "grouping/sa_done_rate_mean": done_mean,
        "grouping/sa_done_rate_p90": done_p90,
        "grouping/sa_done_rate_max": done_max,

        "grouping/ca_bucket_count_mean": bs_mean,
        "grouping/ca_bucket_count_p10": bs_p10,
        "grouping/ca_bucket_count_p50": bs_p50,
        "grouping/ca_bucket_count_p90": bs_p90,
        "grouping/ca_bucket_count_min": bs_min,
        "grouping/ca_bucket_count_max": bs_max,

        "grouping/ca_pairs_observed": float(len(counts)),
    }

    for thr_i, v in frac_ge.items():
        out[f"grouping/trans_frac_in_ca_bucket_ge_{thr_i}"] = float(v)

    # Coverage over latent group-action space. In grid mode this uses the exact effective codebook size K^(Gh*Gw).
    A_obs = int(np.max(act_s) + 1) if act_s.size > 0 else 0
    if A_obs > 0:
        if c.ndim == 1:
            possible_groups = int(codebook_size)
        else:
            possible_groups = _effective_codebook_size(int(codebook_size), (int(c.shape[1]), int(c.shape[2])))
        out["grouping/effective_codebook_size"] = float(possible_groups)
        out["grouping/ca_pairs_possible_in_sample"] = float(possible_groups * A_obs)
        out["grouping/ca_pairs_coverage_frac"] = float(len(counts) / float(possible_groups * A_obs))

    return out


def _dump_top_codes(
    model: VQVAE,
    obs_arr: np.ndarray,
    device: str,
    codebook_size: int,
    eval_sample_size: int,
    topk_print: int,
    examples_per_code: int,
    path: str,
):
    """
    Dumps human-readable examples for top-k codes.
    Works for both single-code and grid-code mode.

    In grid mode, a frame is considered an example of code k if ANY patch uses code k.
    """
    xall = np.asarray(obs_arr, dtype=np.float32)
    if xall.ndim != 4:
        raise ValueError(f"Expected obs_arr shape (N,C,H,W); got {xall.shape}")

    def _ascii_single_frame(obs_chw: np.ndarray) -> str:
        x = np.asarray(obs_chw, dtype=np.float32)
        if x.ndim != 3:
            return "<bad frame>"
        occ = x.sum(axis=0)
        mx = float(np.max(occ)) if occ.size > 0 else 1.0
        if mx <= 0:
            mx = 1.0
        y = occ / mx
        chars = " .:-=+*#%@"
        H, W = y.shape
        out_lines = []
        for r in range(H):
            line = ""
            for c in range(W):
                idx = int(np.clip(round(y[r, c] * (len(chars) - 1)), 0, len(chars) - 1))
                line += chars[idx]
            out_lines.append(line)
        return "\n".join(out_lines)

    n = xall.shape[0]
    m = min(int(eval_sample_size), n)
    idx = np.random.randint(0, n, size=m)
    sample = xall[idx]

    with torch.no_grad():
        codes_t = model.encode_indices(_to_tensor_batch(sample, device=device))
        codes = codes_t.detach().cpu().numpy().astype(np.int64)

    K = int(codebook_size)

    if codes.ndim == 1:
        codes_flat = codes
        frame_has_code = lambda k: (codes == k)
    elif codes.ndim == 3:
        codes_flat = codes.reshape(-1)
        frame_has_code = lambda k: np.any(codes == k, axis=(1, 2))
    else:
        raise ValueError(f"encode_indices returned unexpected shape {codes.shape}")

    counts = np.bincount(codes_flat, minlength=K)
    topk = int(topk_print)
    top_codes = np.argsort(-counts)[:topk]

    lines = []
    lines.append(f"Top-{topk} codes (out of K={K}) from sample m={m}\n")

    for k in top_codes:
        ck = int(counts[k])
        lines.append(f"=== code={k} count={ck} ===")
        if ck <= 0:
            lines.append("<empty>\n")
            continue

        mask = frame_has_code(int(k))
        xb = sample[mask]  # frames containing code k (any patch in grid mode)

        lines.append("[mean map]")
        lines.append(_ascii_mean_map(xb) if xb.shape[0] > 0 else "<no frames>")
        lines.append("")

        ex = min(int(examples_per_code), xb.shape[0])
        if ex > 0:
            pick = np.random.choice(xb.shape[0], size=ex, replace=False)
            for i, pi in enumerate(pick):
                lines.append(f"[example {i+1}/{ex}]")
                lines.append(_ascii_single_frame(xb[int(pi)]))
                lines.append("")
        lines.append("")

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[vqvae] wrote grouping examples -> {path}")


def _load_vqvae_ckpt(load_path: str, device: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(load_path, map_location=device)
    if "state_dict" not in ckpt or "vqvae_cfg" not in ckpt:
        raise ValueError(f"Checkpoint at {load_path} missing state_dict/vqvae_cfg keys.")
    return ckpt, ckpt["vqvae_cfg"]


@hydra.main(config_path="../config", config_name="train_vqvae", version_base=None)
def main(cfg: DictConfig):
    from src.envs.registry import make_env
    from src.utils.wandb_utils import setup_wandb, log_metrics

    # Seed
    set_global_seeds(int(cfg.seed))

    # Device
    device = str(cfg.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # W&B
    run = setup_wandb(cfg, config_dict=OmegaConf.to_container(cfg, resolve=True))

    # Env via registry
    env, eval_env, obs_adapter = make_env(cfg.env, seed=int(cfg.seed))
    print(f"[vqvae] env_id={cfg.env.id}")
    print(f"[vqvae] obs_space={env.observation_space} action_space={env.action_space} n_actions={getattr(env.action_space,'n',None)}")

    obs, _ = env.reset(seed=int(cfg.seed))

    collect_steps = int(cfg.train.collect_steps)
    compute_sa = bool(cfg.diagnostics.compute_sa_stats)

    load_dataset_path = getattr(cfg.outputs, "load_dataset_path", None)
    save_dataset_path = getattr(cfg.outputs, "save_dataset_path", None)
    dataset_only = bool(getattr(cfg.train, "dataset_only", False))

    global_step_start = collect_steps  # training begins after data collection

    obs_list: List[np.ndarray] = []
    trans = {"obs": [], "act": [], "rew": [], "done": [], "next_obs": []}

    # Load or collect data
    if load_dataset_path is not None and str(load_dataset_path) not in ("", "None", "none", "null"):
        obs_arr, transitions_loaded = _load_dataset_npz(str(load_dataset_path))
        if transitions_loaded is not None:
            trans = {k: list(v) for k, v in transitions_loaded.items()}
            compute_sa = True
    else:
        # Collect data
        for t in range(collect_steps):
            a = env.action_space.sample()
            next_obs, r, terminated, truncated, _info = env.step(a)
            d = bool(terminated or truncated)
    
            o_ad = obs_adapter(obs)
            no_ad = obs_adapter(next_obs)
            obs_list.append(o_ad)
    
            if compute_sa:
                trans["obs"].append(o_ad)
                trans["act"].append(int(a))
                trans["rew"].append(float(r))
                trans["done"].append(1.0 if d else 0.0)
                trans["next_obs"].append(no_ad)
    
            obs = next_obs
            if d:
                obs, _ = env.reset()
    
            if (t > 0) and (t % 50_000 == 0):
                log_metrics({"vqvae/collect_step": float(t)}, step=int(t))
    
    env.close()
    if eval_env is not None:
        eval_env.close()

    if load_dataset_path is None or str(load_dataset_path) in ("", "None", "none", "null"):
        obs_arr = np.asarray(obs_list, dtype=np.float32)  # (N,C,H,W)
        transitions_np = None
        if compute_sa:
            transitions_np = {
                "obs": np.asarray(trans["obs"], dtype=np.float32),
                "act": np.asarray(trans["act"], dtype=np.int64),
                "rew": np.asarray(trans["rew"], dtype=np.float32),
                "done": np.asarray(trans["done"], dtype=np.float32),
                "next_obs": np.asarray(trans["next_obs"], dtype=np.float32),
            }
        if save_dataset_path is not None and str(save_dataset_path) not in ("", "None", "none", "null"):
            _save_dataset_npz(str(save_dataset_path), obs_arr=obs_arr, transitions=transitions_np)
        if dataset_only:
            if run is not None:
                run.finish()
            return
    else:
        # already loaded from disk
        if compute_sa and trans:
            transitions_np = {
                "obs": np.asarray(trans["obs"], dtype=np.float32),
                "act": np.asarray(trans["act"], dtype=np.int64),
                "rew": np.asarray(trans["rew"], dtype=np.float32),
                "done": np.asarray(trans["done"], dtype=np.float32),
                "next_obs": np.asarray(trans["next_obs"], dtype=np.float32),
            }
        else:
            transitions_np = None

    obs_arr = np.asarray(obs_arr, dtype=np.float32)
    if obs_arr.ndim != 4:
        raise ValueError(f"Expected collected obs shape (N,C,H,W); got {obs_arr.shape}")

    # -------------------------
    # Build model
    # -------------------------
    codebook_size = int(cfg.vqvae.codebook_size)
    embed_dim = int(cfg.vqvae.embed_dim)
    hidden_channels = int(cfg.vqvae.hidden_channels)
    beta = float(cfg.vqvae.beta)

    grid_size = _parse_grid_size(OmegaConf.select(cfg, "vqvae.grid_size", default=1))

    obs_shape = tuple(obs_arr.shape[1:])  # (C,H,W)
    in_channels = int(obs_arr.shape[1])

    eval_only = bool(getattr(cfg.train, "eval_only", False))
    load_path = getattr(cfg.outputs, "load_path", None)

    if eval_only:
        if load_path is None or str(load_path) == "" or str(load_path).lower() == "none":
            raise ValueError("train.eval_only=true requires outputs.load_path to point to a saved vqvae checkpoint.")

        ckpt, vcfg = _load_vqvae_ckpt(str(load_path), device=device)

        grid_size = _parse_grid_size(vcfg.get("grid_size", 1))

        model = VQVAE(
            in_channels=int(vcfg["in_channels"]),
            obs_shape=tuple(vcfg["obs_shape"]),
            codebook_size=int(vcfg["codebook_size"]),
            embed_dim=int(vcfg["embed_dim"]),
            hidden_channels=int(vcfg["hidden_channels"]),
            beta=float(vcfg["beta"]),
            grid_size=grid_size,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        codebook_size = int(vcfg["codebook_size"])
        train_steps = 0
        global_step_end = int(collect_steps + train_steps)
        print(f"[vqvae] eval-only: loaded {load_path} (K={codebook_size}, grid={grid_size})")

    else:
        model = VQVAE(
            in_channels=in_channels,
            obs_shape=obs_shape,
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            hidden_channels=hidden_channels,
            beta=beta,
            grid_size=grid_size,
        ).to(device)

        opt = optim.Adam(model.parameters(), lr=float(cfg.train.lr))

        n = obs_arr.shape[0]
        bs = int(cfg.train.batch_size)
        train_steps = int(cfg.train.train_steps)
        log_every = int(cfg.train.log_interval_steps)
        recon_loss = str(cfg.train.recon_loss).lower()

        for step in range(train_steps):
            idxb = np.random.randint(0, n, size=bs)
            batch = obs_arr[idxb]
            x = _to_tensor_batch(batch, device=device)

            out = model(x)
            x_hat = out["x_hat"]
            vq_loss = out["vq_loss"]

            if recon_loss == "bce":
                x_in = x.clamp(0.0, 1.0)
                xh = torch.sigmoid(x_hat)
                recon = torch.mean(F.binary_cross_entropy(xh, x_in, reduction="none"))
            elif recon_loss == "mse":
                recon = torch.mean((x_hat - x) ** 2)
            else:
                raise ValueError(f"Unknown recon_loss={recon_loss}")

            loss = recon + vq_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (step % log_every) == 0:
                gstep = global_step_start + step
                metrics = {
                    "vqvae/train_loss": float(loss.item()),
                    "vqvae/recon_loss": float(recon.item()),
                    "vqvae/vq_loss": float(vq_loss.item()),
                    "vqvae/train_step": float(step),
                    "vqvae/grid_h": float(grid_size[0]),
                    "vqvae/grid_w": float(grid_size[1]),
                }
                log_metrics(metrics, step=int(gstep))
                print(
                    f"[vqvae] step={step} gstep={gstep} "
                    f"loss={metrics['vqvae/train_loss']:.6f} "
                    f"recon={metrics['vqvae/recon_loss']:.6f} "
                    f"vq={metrics['vqvae/vq_loss']:.6f} "
                    f"grid={grid_size}"
                )

        global_step_end = int(collect_steps + train_steps)

    # -------------------------
    # Save checkpoint
    # -------------------------
    if not eval_only:
        save_path = str(cfg.outputs.save_path)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        ckpt = {
            "state_dict": model.state_dict(),
            "obs_shape": obs_shape,
            "vqvae_cfg": {
                "in_channels": in_channels,
                "obs_shape": obs_shape,
                "codebook_size": codebook_size,
                "embed_dim": embed_dim,
                "hidden_channels": hidden_channels,
                "beta": beta,
                "grid_size": list(grid_size),
            },
            "env_cfg": OmegaConf.to_container(cfg.env, resolve=True),
            "global_step_end": int(global_step_end),
        }
        torch.save(ckpt, save_path)
        print(f"[vqvae] saved -> {save_path}")

    # -------------------------
    # Diagnostics
    # -------------------------
    model.eval()

    eval_sample_size = int(cfg.diagnostics.eval_sample_size)
    code_metrics = _compute_code_metrics(
        model=model,
        obs_arr=obs_arr,
        device=device,
        codebook_size=codebook_size,
        eval_sample_size=eval_sample_size,
    )
    log_metrics(code_metrics, step=int(global_step_end))

    if compute_sa and transitions_np is not None:
        bucket_thresholds = list(OmegaConf.select(cfg, "diagnostics.bucket_thresholds", default=[5, 10, 50, 100]))
        sa_metrics = _compute_sa_stats(
            model=model,
            transitions=transitions_np,
            device=device,
            codebook_size=codebook_size,
            sa_eval_sample_size=int(cfg.diagnostics.sa_eval_sample_size),
            min_count_for_sa=int(cfg.diagnostics.min_count_for_sa),
            bucket_thresholds=[int(x) for x in bucket_thresholds],
        )
        log_metrics(sa_metrics, step=int(global_step_end))

    dump_path = str(cfg.outputs.dump_examples_path) if cfg.outputs.dump_examples_path is not None else ""
    if dump_path:
        examples_per_code = int(OmegaConf.select(cfg, "diagnostics.examples_per_code", default=4))
        _dump_top_codes(
            model=model,
            obs_arr=obs_arr,
            device=device,
            codebook_size=codebook_size,
            eval_sample_size=eval_sample_size,
            topk_print=int(cfg.diagnostics.topk_print),
            examples_per_code=examples_per_code,
            path=dump_path,
        )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
