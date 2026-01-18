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


def _compute_code_metrics(model: VQVAE, obs_arr: np.ndarray, device: str, codebook_size: int, eval_sample_size: int) -> Dict[str, float]:
    """
    Computes codebook usage, perplexity, and intra-code variance.
    Assumes obs_arr is (N,C,H,W).
    """
    xall = np.asarray(obs_arr, dtype=np.float32)
    if xall.ndim != 4:
        raise ValueError(f"Expected obs_arr shape (N,C,H,W); got {xall.shape}")

    n = xall.shape[0]
    m = min(int(eval_sample_size), n)
    idx = np.random.randint(0, n, size=m)
    sample = xall[idx]  # (m,C,H,W)

    with torch.no_grad():
        codes = model.encode_indices(_to_tensor_batch(sample, device=device)).detach().cpu().numpy().astype(np.int64)

    K = int(codebook_size)
    counts = np.bincount(codes, minlength=K).astype(np.int64)

    used = int(np.sum(counts > 0))
    usage_frac = float(used) / float(K)

    H = _entropy_from_counts(counts)
    perplexity = float(np.exp(H))

    # intra-code variance proxy (binary-ish):
    # var proxy = mean_{pixels}(mu*(1-mu)) for each code
    intra_vars = []
    for k in range(K):
        ck = int(counts[k])
        if ck <= 0:
            continue
        mask = (codes == k)
        xb = sample[mask]               # (ck,C,H,W)
        mu_k = xb.mean(axis=0)          # (C,H,W)
        var_k = float(np.mean(mu_k * (1.0 - mu_k)))
        intra_vars.append(var_k)

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


def _compute_sa_stats(
    model: VQVAE,
    transitions: Dict[str, np.ndarray],
    device: str,
    codebook_size: int,
    sa_eval_sample_size: int,
    min_count_for_sa: int,
) -> Dict[str, float]:
    """
    Computes (code,action)-level diagnostics:
      - reward std within (c,a)
      - done rate within (c,a)
      - entropy of next_code within (c,a)
    Assumes obs and next_obs are (N,C,H,W).
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
        c = model.encode_indices(_to_tensor_batch(obs_s, device=device)).cpu().numpy().astype(np.int64)
        c2 = model.encode_indices(_to_tensor_batch(next_s, device=device)).cpu().numpy().astype(np.int64)

    K = int(codebook_size)

    from collections import defaultdict
    counts = defaultdict(int)
    rew_sum = defaultdict(float)
    rew_sumsq = defaultdict(float)
    done_sum = defaultdict(float)
    next_counts = defaultdict(lambda: np.zeros((K,), dtype=np.int64))

    for ci, ai, ri, di, c2i in zip(c, act_s, rew_s, done_s, c2):
        key = (int(ci), int(ai))
        counts[key] += 1
        rew_sum[key] += float(ri)
        rew_sumsq[key] += float(ri * ri)
        done_sum[key] += float(di)
        next_counts[key][int(c2i)] += 1

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
        ent = _entropy_from_counts(next_counts[key])
        ent_list.append(ent)
        rstd_list.append(rstd)
        done_list.append(dr)

    return {
        "grouping/sa_pairs_counted": float(len(ent_list)),
        "grouping/sa_next_code_entropy_mean": float(np.mean(ent_list)) if ent_list else 0.0,
        "grouping/sa_reward_std_mean": float(np.mean(rstd_list)) if rstd_list else 0.0,
        "grouping/sa_done_rate_mean": float(np.mean(done_list)) if done_list else 0.0,
    }


def _dump_top_codes(model: VQVAE, obs_arr: np.ndarray, device: str, codebook_size: int, eval_sample_size: int, topk_print: int, path: str):
    """
    Dumps human-readable examples for top-k codes:
      - count
      - ASCII mean map
    Assumes obs_arr is (N,C,H,W).
    """
    xall = np.asarray(obs_arr, dtype=np.float32)
    if xall.ndim != 4:
        raise ValueError(f"Expected obs_arr shape (N,C,H,W); got {xall.shape}")

    n = xall.shape[0]
    m = min(int(eval_sample_size), n)
    idx = np.random.randint(0, n, size=m)
    sample = xall[idx]

    with torch.no_grad():
        codes = model.encode_indices(_to_tensor_batch(sample, device=device)).cpu().numpy().astype(np.int64)

    K = int(codebook_size)
    counts = np.bincount(codes, minlength=K)

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
        mask = (codes == k)
        xb = sample[mask]  # (ck,C,H,W)
        lines.append(_ascii_mean_map(xb))
        lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[vqvae] wrote grouping examples -> {path}")


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

    # Env via registry (ensures sticky actions, adapters, wrappers match RL)
    env, eval_env, obs_adapter = make_env(cfg.env, seed=int(cfg.seed))
    print(f"[vqvae] env_id={cfg.env.id}")
    print(f"[vqvae] obs_space={env.observation_space} action_space={env.action_space} n_actions={getattr(env.action_space,'n',None)}")

    obs, _ = env.reset(seed=int(cfg.seed))

    collect_steps = int(cfg.train.collect_steps)
    compute_sa = bool(cfg.diagnostics.compute_sa_stats)

    obs_list = []
    trans = {"obs": [], "act": [], "rew": [], "done": [], "next_obs": []}

    for t in range(collect_steps):
        a = env.action_space.sample()
        next_obs, r, terminated, truncated, _info = env.step(a)
        d = bool(terminated or truncated)

        # Store ADAPTED obs: (C,H,W) float32
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

        # lightweight logging during collection
        if (t > 0) and (t % 50000 == 0):
            log_metrics({"vqvae/collect_step": float(t)}, step=t)

    env.close()
    eval_env.close()

    obs_arr = np.asarray(obs_list, dtype=np.float32)  # (N,C,H,W)
    if obs_arr.ndim != 4:
        raise ValueError(f"Expected collected obs shape (N,C,H,W); got {obs_arr.shape}")

    # Model config
    codebook_size = int(cfg.vqvae.codebook_size)
    embed_dim = int(cfg.vqvae.embed_dim)
    hidden_channels = int(cfg.vqvae.hidden_channels)
    beta = float(cfg.vqvae.beta)

    # Derived shapes
    obs_shape = tuple(obs_arr.shape[1:])  # (C,H,W)
    in_channels = int(obs_arr.shape[1])

    model = VQVAE(
        in_channels=in_channels,
        obs_shape=obs_shape,
        codebook_size=codebook_size,
        embed_dim=embed_dim,
        hidden_channels=hidden_channels,
        beta=beta,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=float(cfg.train.lr))

    # Training params
    n = obs_arr.shape[0]
    bs = int(cfg.train.batch_size)
    train_steps = int(cfg.train.train_steps)
    log_every = int(cfg.train.log_interval_steps)
    recon_loss = str(cfg.train.recon_loss).lower()

    for step in range(train_steps):
        idx = np.random.randint(0, n, size=bs)
        batch = obs_arr[idx]  # (B,C,H,W)
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
            metrics = {
                "vqvae/train_loss": float(loss.item()),
                "vqvae/recon_loss": float(recon.item()),
                "vqvae/vq_loss": float(vq_loss.item()),
                "vqvae/step": float(step),
            }
            log_metrics(metrics, step=step)
            print(f"[vqvae] step={step} loss={metrics['vqvae/train_loss']:.6f} recon={metrics['vqvae/recon_loss']:.6f} vq={metrics['vqvae/vq_loss']:.6f}")

    # Save checkpoint
    save_path = str(cfg.outputs.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
        },
        "env_cfg": OmegaConf.to_container(cfg.env, resolve=True),
    }
    torch.save(ckpt, save_path)
    print(f"[vqvae] saved -> {save_path}")

    # Diagnostics (log + print)
    model.eval()

    eval_sample_size = int(cfg.diagnostics.eval_sample_size)
    code_metrics = _compute_code_metrics(
        model=model,
        obs_arr=obs_arr,
        device=device,
        codebook_size=codebook_size,
        eval_sample_size=eval_sample_size,
    )
    log_metrics(code_metrics, step=train_steps)

    if compute_sa:
        transitions = {
            "obs": np.asarray(trans["obs"], dtype=np.float32),
            "act": np.asarray(trans["act"], dtype=np.int64),
            "rew": np.asarray(trans["rew"], dtype=np.float32),
            "done": np.asarray(trans["done"], dtype=np.float32),
            "next_obs": np.asarray(trans["next_obs"], dtype=np.float32),
        }
        sa_metrics = _compute_sa_stats(
            model=model,
            transitions=transitions,
            device=device,
            codebook_size=codebook_size,
            sa_eval_sample_size=int(cfg.diagnostics.sa_eval_sample_size),
            min_count_for_sa=int(cfg.diagnostics.min_count_for_sa),
        )
        log_metrics(sa_metrics, step=train_steps)

    dump_path = str(cfg.outputs.dump_examples_path) if cfg.outputs.dump_examples_path is not None else ""
    if dump_path:
        _dump_top_codes(
            model=model,
            obs_arr=obs_arr,
            device=device,
            codebook_size=codebook_size,
            eval_sample_size=eval_sample_size,
            topk_print=int(cfg.diagnostics.topk_print),
            path=dump_path,
        )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
