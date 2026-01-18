import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List, DefaultDict
from collections import defaultdict

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym

from src.models.vqvae import VQVAE
from src.utils.seed import set_global_seeds


@dataclass
class VQVAETrainCfg:
    # data collection
    env_id: str = "MinAtar/Breakout-v0"
    seed: int = 0
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
    device: str = "cpu"  # set "cuda" if available
    dump_examples_path: Optional[str] = "artifacts/vqvae_group_examples.txt"


def _to_tensor_batch(obs_batch: np.ndarray, device: str) -> torch.Tensor:
    x = np.asarray(obs_batch)
    if x.ndim == 4:
        # (B,H,W,C) -> (B,C,H,W)
        x = x.astype(np.float32)
        x = np.transpose(x, (0, 3, 1, 2))
        return torch.from_numpy(x).to(device)
    if x.ndim == 2:
        return torch.from_numpy(x.astype(np.float32)).to(device)
    raise ValueError(f"Unsupported batch shape {x.shape}")


def _entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    c = counts.astype(np.float64)
    s = float(c.sum())
    if s <= 0:
        return 0.0
    p = c / s
    return float(-(p * np.log(p + eps)).sum())


def _ascii_mean_map(obs_batch_hwc: np.ndarray) -> str:
    """
    obs_batch_hwc: (N,H,W,C) binary-ish.
    Show mean occupancy over channels as ASCII.
    """
    if obs_batch_hwc.ndim != 4:
        return "<non-image obs>"
    mu = obs_batch_hwc.mean(axis=0)  # (H,W,C)
    occ = mu.sum(axis=2)            # (H,W)
    # normalize for display
    mx = float(np.max(occ)) if occ.size > 0 else 1.0
    if mx <= 0:
        mx = 1.0
    x = occ / mx
    chars = " .:-=+*#%@"
    out_lines = []
    for r in range(x.shape[0]):
        line = ""
        for c in range(x.shape[1]):
            idx = int(np.clip(round(x[r, c] * (len(chars) - 1)), 0, len(chars) - 1))
            line += chars[idx]
        out_lines.append(line)
    return "\n".join(out_lines)


def _compute_code_metrics(model: VQVAE, obs_arr: np.ndarray, cfg: VQVAETrainCfg) -> Dict[str, float]:
    """
    Computes codebook usage, perplexity, and intra-code variance.
    """
    device = cfg.device
    n = obs_arr.shape[0]
    m = min(int(cfg.eval_sample_size), n)
    idx = np.random.randint(0, n, size=m)
    sample = obs_arr[idx]

    # encode -> codes
    with torch.no_grad():
        x = _to_tensor_batch(sample, device=device)
        codes = model.encode_indices(x).detach().cpu().numpy().astype(np.int64)

    K = int(cfg.codebook_size)
    counts = np.bincount(codes, minlength=K).astype(np.int64)

    used = int(np.sum(counts > 0))
    usage_frac = float(used) / float(K)

    H = _entropy_from_counts(counts)
    perplexity = float(np.exp(H))

    # intra-code variance proxy for binary obs:
    # for each code: mu = mean(obs), var = mean(mu*(1-mu)) averaged over pixels+channels
    # then average var across codes that appear
    intra_vars = []
    if sample.ndim == 4:
        # (m,H,W,C)
        for k in range(K):
            ck = counts[k]
            if ck <= 0:
                continue
            mask = (codes == k)
            xb = sample[mask].astype(np.float32)  # (ck,H,W,C)
            mu = xb.mean(axis=0)
            var = float(np.mean(mu * (1.0 - mu)))
            intra_vars.append(var)

    intra_mean = float(np.mean(intra_vars)) if intra_vars else 0.0
    intra_p90 = float(np.quantile(np.asarray(intra_vars), 0.90)) if len(intra_vars) >= 5 else intra_mean

    return {
        "vqvae/code_usage_frac": usage_frac,
        "vqvae/code_used": float(used),
        "vqvae/code_entropy": float(H),
        "vqvae/perplexity": float(perplexity),
        "vqvae/intra_code_var_mean": float(intra_mean),
        "vqvae/intra_code_var_p90": float(intra_p90),
    }


def _compute_sa_stats(model: VQVAE, transitions: Dict[str, np.ndarray], cfg: VQVAETrainCfg) -> Dict[str, float]:
    """
    Computes (code,action)-level diagnostics:
      - reward std within (c,a)
      - done rate within (c,a)
      - entropy of next_code within (c,a)
    """
    device = cfg.device
    obs = transitions["obs"]
    act = transitions["act"].astype(np.int64)
    rew = transitions["rew"].astype(np.float32)
    done = transitions["done"].astype(np.float32)
    next_obs = transitions["next_obs"]

    n = obs.shape[0]
    m = min(int(cfg.sa_eval_sample_size), n)
    idx = np.random.randint(0, n, size=m)

    obs_s = obs[idx]
    act_s = act[idx]
    rew_s = rew[idx]
    done_s = done[idx]
    next_s = next_obs[idx]

    with torch.no_grad():
        c = model.encode_indices(_to_tensor_batch(obs_s, device=device)).cpu().numpy().astype(np.int64)
        c2 = model.encode_indices(_to_tensor_batch(next_s, device=device)).cpu().numpy().astype(np.int64)

    # aggregate per (c,a)
    K = int(cfg.codebook_size)
    A = int(np.max(act_s) + 1)

    # counts[(c,a)] and next_code_counts[(c,a)] for entropy
    counts = defaultdict(int)
    rew_sum = defaultdict(float)
    rew_sumsq = defaultdict(float)
    done_sum = defaultdict(float)
    next_counts: DefaultDict[Tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros((K,), dtype=np.int64))

    for ci, ai, ri, di, c2i in zip(c, act_s, rew_s, done_s, c2):
        key = (int(ci), int(ai))
        counts[key] += 1
        rew_sum[key] += float(ri)
        rew_sumsq[key] += float(ri * ri)
        done_sum[key] += float(di)
        next_counts[key][int(c2i)] += 1

    # summarize over keys with enough support
    ent_list = []
    rstd_list = []
    done_list = []
    for key, cnt in counts.items():
        if cnt < int(cfg.min_count_for_sa):
            continue
        mu = rew_sum[key] / float(cnt)
        var = max(0.0, (rew_sumsq[key] / float(cnt)) - mu * mu)
        rstd = float(np.sqrt(var))
        dr = done_sum[key] / float(cnt)
        ent = _entropy_from_counts(next_counts[key])
        ent_list.append(ent)
        rstd_list.append(rstd)
        done_list.append(dr)

    out = {
        "grouping/sa_pairs_counted": float(len(ent_list)),
        "grouping/sa_next_code_entropy_mean": float(np.mean(ent_list)) if ent_list else 0.0,
        "grouping/sa_reward_std_mean": float(np.mean(rstd_list)) if rstd_list else 0.0,
        "grouping/sa_done_rate_mean": float(np.mean(done_list)) if done_list else 0.0,
    }
    return out


def _dump_top_codes(model: VQVAE, obs_arr: np.ndarray, cfg: VQVAETrainCfg, path: str):
    """
    Dumps human-readable examples for top-k codes:
      - count
      - ASCII mean map
    """
    device = cfg.device
    n = obs_arr.shape[0]
    m = min(int(cfg.eval_sample_size), n)
    idx = np.random.randint(0, n, size=m)
    sample = obs_arr[idx]

    with torch.no_grad():
        codes = model.encode_indices(_to_tensor_batch(sample, device=device)).cpu().numpy().astype(np.int64)

    K = int(cfg.codebook_size)
    counts = np.bincount(codes, minlength=K)

    topk = int(cfg.topk_print)
    top_codes = np.argsort(-counts)[:topk]

    lines = []
    lines.append(f"Top-{topk} codes (out of K={K}) from sample m={m}\n")
    for k in top_codes:
        ck = int(counts[k])
        lines.append(f"=== code={k} count={ck} ===")
        if ck <= 0:
            lines.append("<empty>\n")
            continue
        mask = (codes == k)
        xb = sample[mask]
        if xb.ndim == 4:
            lines.append(_ascii_mean_map(xb))
        else:
            lines.append(f"<non-image obs shape={xb.shape}>")
        lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[vqvae] wrote grouping examples -> {path}")


@hydra.main(config_path="../config", config_name="train_vqvae", version_base=None)
def main(cfg: DictConfig):
    cfg_py = OmegaConf.to_container(cfg, resolve=True)
    cfg2 = VQVAETrainCfg(**cfg_py)

    os.makedirs(os.path.dirname(cfg2.save_path), exist_ok=True)
    set_global_seeds(cfg2.seed)

    device = cfg2.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg2.device = device

    env = gym.make(cfg2.env_id)
    print(f"[vqvae] env_id={cfg2.env_id}")
    print(f"[vqvae] obs_space={env.observation_space} action_space={env.action_space} n_actions={getattr(env.action_space,'n',None)}")

    obs, _ = env.reset(seed=cfg2.seed)

    # collect observations AND transitions (for (code,action) diagnostics)
    obs_list = []
    trans = {
        "obs": [],
        "act": [],
        "rew": [],
        "done": [],
        "next_obs": [],
    }

    for t in range(int(cfg2.collect_steps)):
        a = env.action_space.sample()
        next_obs, r, terminated, truncated, _info = env.step(a)
        d = bool(terminated or truncated)

        obs_list.append(obs)

        if cfg2.compute_sa_stats:
            trans["obs"].append(obs)
            trans["act"].append(int(a))
            trans["rew"].append(float(r))
            trans["done"].append(1.0 if d else 0.0)
            trans["next_obs"].append(next_obs)

        obs = next_obs
        if d:
            obs, _ = env.reset()

    env.close()

    obs_arr = np.asarray(obs_list)

    if obs_arr.ndim == 4:
        # (N,H,W,C) -> model expects CHW shape for obs_shape
        H, W, C = obs_arr.shape[1], obs_arr.shape[2], obs_arr.shape[3]
        obs_shape = (C, H, W)
        in_channels = C
    elif obs_arr.ndim == 2:
        D = obs_arr.shape[1]
        obs_shape = (D,)
        in_channels = 1
    else:
        raise ValueError(f"Unexpected collected obs shape: {obs_arr.shape}")

    model = VQVAE(
        in_channels=in_channels,
        obs_shape=obs_shape,
        codebook_size=cfg2.codebook_size,
        embed_dim=cfg2.embed_dim,
        hidden_channels=cfg2.hidden_channels,
        beta=cfg2.beta,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg2.lr)

    n = obs_arr.shape[0]
    bs = int(cfg2.batch_size)

    # training loop
    for step in range(int(cfg2.train_steps)):
        idx = np.random.randint(0, n, size=bs)
        batch = obs_arr[idx]
        x = _to_tensor_batch(batch, device=device)

        out = model(x)
        x_hat = out["x_hat"]
        vq_loss = out["vq_loss"]

        if cfg2.recon_loss == "bce":
            # MinAtar is binary-ish; clamp for safety
            x_in = x.clamp(0.0, 1.0)
            xh = torch.sigmoid(x_hat)
            recon = torch.mean(F.binary_cross_entropy(xh, x_in, reduction="none"))
        else:
            recon = torch.mean((x_hat - x) ** 2)

        loss = recon + vq_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 1000 == 0:
            print(
                f"[vqvae] step={step} loss={float(loss.item()):.6f} "
                f"recon={float(recon.item()):.6f} vq={float(vq_loss.item()):.6f}"
            )

    # save
    ckpt = {
        "state_dict": model.state_dict(),
        "obs_shape": obs_shape,
        "vqvae_cfg": {
            "in_channels": in_channels,
            "obs_shape": obs_shape,
            "codebook_size": cfg2.codebook_size,
            "embed_dim": cfg2.embed_dim,
            "hidden_channels": cfg2.hidden_channels,
            "beta": cfg2.beta,
        },
    }
    torch.save(ckpt, cfg2.save_path)
    print(f"[vqvae] saved -> {cfg2.save_path}")

    # diagnostics on grouping quality
    model.eval()
    code_metrics = _compute_code_metrics(model, obs_arr, cfg2)
    print("[vqvae] codebook diagnostics:")
    for k, v in code_metrics.items():
        print(f"  {k}: {v}")

    if cfg2.compute_sa_stats:
        transitions = {
            "obs": np.asarray(trans["obs"]),
            "act": np.asarray(trans["act"]),
            "rew": np.asarray(trans["rew"], dtype=np.float32),
            "done": np.asarray(trans["done"], dtype=np.float32),
            "next_obs": np.asarray(trans["next_obs"]),
        }
        sa_metrics = _compute_sa_stats(model, transitions, cfg2)
        print("[vqvae] (code,action) diagnostics:")
        for k, v in sa_metrics.items():
            print(f"  {k}: {v}")

    if cfg2.dump_examples_path:
        _dump_top_codes(model, obs_arr, cfg2, cfg2.dump_examples_path)


if __name__ == "__main__":
    main()
