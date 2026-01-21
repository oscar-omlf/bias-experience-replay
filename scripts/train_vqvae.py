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
    bucket_thresholds: List[int],
) -> Dict[str, float]:
    """
    Computes (code,action)-level diagnostics for latent grouping quality.

    Returns metrics including:
      - counts distribution over observed (code,action) buckets
      - fraction of transitions that fall into buckets with count >= threshold
      - reward std within (code,action)
      - done rate within (code,action)
      - entropy of next_code within (code,action)

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

    # Aggregate per (c,a)
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

    # Bucket size distribution over *all observed* (c,a)
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

    # Fraction of transitions that are in “usable” buckets (based on bucket count)
    per_transition_bucket_size = np.asarray(
        [counts[(int(ci), int(ai))] for ci, ai in zip(c, act_s)],
        dtype=np.int64,
    )

    frac_ge = {}
    for thr in bucket_thresholds:
        thr_i = int(thr)
        frac_ge[thr_i] = float(np.mean(per_transition_bucket_size >= thr_i))

    # Summarize per (c,a) keys with enough support (>= min_count_for_sa)
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
    }

    for thr_i, v in frac_ge.items():
        out[f"grouping/trans_frac_in_ca_bucket_ge_{thr_i}"] = float(v)

    A_obs = int(np.max(act_s) + 1) if act_s.size > 0 else 0
    out["grouping/ca_pairs_observed"] = float(len(counts))
    out["grouping/ca_pairs_possible_in_sample"] = float(K * A_obs) if A_obs > 0 else 0.0
    out["grouping/ca_pairs_coverage_frac"] = float(len(counts) / float(K * A_obs)) if A_obs > 0 else 0.0

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
    Dumps human-readable examples for top-k codes:
      - count
      - ASCII mean map
      - a few individual example frames

    Assumes obs_arr is (N,C,H,W).
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

        lines.append("[mean map]")
        lines.append(_ascii_mean_map(xb))
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

    # Global-step anchors (FIX: avoids NameError and keeps logs consistent)
    global_step_start = collect_steps  # training begins after data collection

    obs_list: List[np.ndarray] = []
    trans = {"obs": [], "act": [], "rew": [], "done": [], "next_obs": []}

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

    obs_arr = np.asarray(obs_list, dtype=np.float32)  # (N,C,H,W)
    if obs_arr.ndim != 4:
        raise ValueError(f"Expected collected obs shape (N,C,H,W); got {obs_arr.shape}")

    # -------------------------
    # Build model
    # -------------------------
    codebook_size = int(cfg.vqvae.codebook_size)
    embed_dim = int(cfg.vqvae.embed_dim)
    hidden_channels = int(cfg.vqvae.hidden_channels)
    beta = float(cfg.vqvae.beta)

    obs_shape = tuple(obs_arr.shape[1:])  # (C,H,W)
    in_channels = int(obs_arr.shape[1])

    eval_only = bool(getattr(cfg.train, "eval_only", False))
    load_path = getattr(cfg.outputs, "load_path", None)

    if eval_only:
        if load_path is None or str(load_path) == "" or str(load_path).lower() == "none":
            raise ValueError("train.eval_only=true requires outputs.load_path to point to a saved vqvae checkpoint.")

        ckpt, vcfg = _load_vqvae_ckpt(str(load_path), device=device)

        model = VQVAE(
            in_channels=int(vcfg["in_channels"]),
            obs_shape=tuple(vcfg["obs_shape"]),
            codebook_size=int(vcfg["codebook_size"]),
            embed_dim=int(vcfg["embed_dim"]),
            hidden_channels=int(vcfg["hidden_channels"]),
            beta=float(vcfg["beta"]),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        codebook_size = int(vcfg["codebook_size"])
        train_steps = 0
        global_step_end = int(collect_steps + train_steps)  # FIX: define early for downstream use
        print(f"[vqvae] eval-only: loaded {load_path} (K={codebook_size})")

    else:
        model = VQVAE(
            in_channels=in_channels,
            obs_shape=obs_shape,
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            hidden_channels=hidden_channels,
            beta=beta,
        ).to(device)

        opt = optim.Adam(model.parameters(), lr=float(cfg.train.lr))

        n = obs_arr.shape[0]
        bs = int(cfg.train.batch_size)
        train_steps = int(cfg.train.train_steps)
        log_every = int(cfg.train.log_interval_steps)
        recon_loss = str(cfg.train.recon_loss).lower()

        for step in range(train_steps):
            idx = np.random.randint(0, n, size=bs)
            batch = obs_arr[idx]
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
                gstep = global_step_start + step  # FIX: now defined
                metrics = {
                    "vqvae/train_loss": float(loss.item()),
                    "vqvae/recon_loss": float(recon.item()),
                    "vqvae/vq_loss": float(vq_loss.item()),
                    "vqvae/train_step": float(step),
                }
                log_metrics(metrics, step=int(gstep))
                print(
                    f"[vqvae] step={step} gstep={gstep} "
                    f"loss={metrics['vqvae/train_loss']:.6f} "
                    f"recon={metrics['vqvae/recon_loss']:.6f} "
                    f"vq={metrics['vqvae/vq_loss']:.6f}"
                )

        global_step_end = int(collect_steps + train_steps)  # FIX: defined before saving/diagnostics

    # -------------------------
    # Save checkpoint
    # -------------------------
    if not eval_only:
        save_path = str(cfg.outputs.save_path)
        save_dir = os.path.dirname(save_path)
        if save_dir:  # FIX: avoids os.makedirs("") crash when save_path has no directory
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
            },
            "env_cfg": OmegaConf.to_container(cfg.env, resolve=True),
            "global_step_end": int(global_step_end),  # FIX: now always defined
        }
        torch.save(ckpt, save_path)
        print(f"[vqvae] saved -> {save_path}")

    # -------------------------
    # Diagnostics (log + print)
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

    if compute_sa:
        transitions = {
            "obs": np.asarray(trans["obs"], dtype=np.float32),
            "act": np.asarray(trans["act"], dtype=np.int64),
            "rew": np.asarray(trans["rew"], dtype=np.float32),
            "done": np.asarray(trans["done"], dtype=np.float32),
            "next_obs": np.asarray(trans["next_obs"], dtype=np.float32),
        }

        bucket_thresholds = list(OmegaConf.select(cfg, "diagnostics.bucket_thresholds", default=[5, 10, 50, 100]))
        sa_metrics = _compute_sa_stats(
            model=model,
            transitions=transitions,
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
