from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random
import numpy as np
import torch
import torch.optim as optim
import wandb

from src.replay import make_replay
from src.models.factory import build_q_network
from src.models.tabular_model import TabularDynamicsModel
from src.utils.schedules import LinearSchedule
from src.utils.wandb_utils import log_metrics
from src.algo.dqn import DoubleDQN


@dataclass
class EpisodeStats:
    reward_sum: float = 0.0
    length: int = 0
    success: int = 0


class DQNAgent:
    def __init__(self, cfg, env, eval_env, obs_adapter, device: str = "cpu"):
        self.cfg = cfg
        self.env = env
        self.eval_env = eval_env
        self.obs_adapter = obs_adapter
        self.device = device
        self.target_updates = 0

        self.step_logs = []
        self.episode_logs = []
        self.eval_logs = []

        obs_space = env.observation_space
        action_space = env.action_space
        assert hasattr(action_space, "n"), "This baseline expects a discrete action space."

        # Input dim (use adapter for FrozenLake)
        if hasattr(obs_space, "n"):  # Discrete
            self.input_dim = obs_space.n
        else:
            self.input_dim = int(np.prod(obs_space.shape))

        self.n_actions = action_space.n
        self.model_type = str(self.cfg.agents.model.type)

        # Networks
        self.q_net, info = build_q_network(self.cfg.agents.model, obs_space, self.n_actions)
        self.q_net.to(self.device)

        self.target_q_net, _ = build_q_network(self.cfg.agents.model, obs_space, self.n_actions)
        self.target_q_net.to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self._mlp_input_dim = info.get("input_dim", None)

        print(f"[Init] obs_space={obs_space}, n_actions={self.n_actions}, model={self.model_type}, info={info}")

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=self.cfg.agents.optimizer.lr,
            weight_decay=self.cfg.agents.optimizer.weight_decay
        )

        # Algo
        self.algo = DoubleDQN(
            q_net=self.q_net,
            target_q_net=self.target_q_net,
            gamma=self.cfg.agents.gamma,
            max_grad_norm=self.cfg.agents.max_grad_norm,
            device=self.device,
            handle_time_limit_as_terminal=self.cfg.agents.handle_time_limit_as_terminal,
        )

        # Replay
        self.replay = make_replay(self.cfg, obs_space=obs_space, device=self.device)

        # Epsilon schedule
        self.eps_sched = LinearSchedule(
            start=self.cfg.agents.epsilon.start,
            end=self.cfg.agents.epsilon.end,
            duration=self.cfg.agents.epsilon.decay_steps,
        )

        self.global_step = 0
        self.is_per = str(self.cfg.agents.replay.type).lower() == "per"
        self.mit_cfg = getattr(self.cfg.agents.replay, "sa_mitigation", None)

        print(f"[Init] Loaded {self.cfg.agents.algo} agent with {self.cfg.agents.replay.type} replay.")

        # Optional tabular model for MODEL mitigation
        self.env_model = None
        if self.is_per and self.mit_cfg is not None and bool(self.mit_cfg.enabled):
            method = str(self.mit_cfg.method).lower()
            if method == "model":
                if not hasattr(obs_space, "n") or not hasattr(action_space, "n"):
                    raise ValueError("MODEL mitigation needs discrete observation and action space.")
                self.env_model = TabularDynamicsModel(
                    n_states=obs_space.n,
                    n_actions=action_space.n,
                )
                print("[Init] TabularDynamicsModel initialized for model-based PER mitigation.")

        # per-action stats (policy & replay) and bandit detection
        self.action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self.sample_counts = np.zeros(self.n_actions, dtype=np.int64)

        # Bandit-like env exposes true_means at the unwrapped level
        self._is_bandit_env = hasattr(self.env.unwrapped, "true_means")

        # Logging guard to avoid W&B spam on large action spaces
        self._max_action_frac_logs = int(getattr(getattr(self.cfg.agents, "logging", None), "max_action_frac_logs", 20))

        # Extra evaluation / diagnostic metrics
        self.first_train_goal_step: Optional[int] = None
        self.first_eval_goal_step: Optional[int] = None

        self._eval_auc_success = 0.0
        self._eval_auc_return = 0.0
        self._last_eval_step: Optional[int] = None
        self._last_eval_success: Optional[float] = None
        self._last_eval_return: Optional[float] = None

        # Bandit AUC diagnostics (for conal/bandit envs)
        self._bandit_auc_mse = 0.0
        self._bandit_auc_mae = 0.0
        self._bandit_auc_q_range = 0.0
        self._bandit_auc_greedy_sigma = 0.0

        self._last_bandit_step: Optional[int] = None
        self._last_bandit_mse: Optional[float] = None
        self._last_bandit_mae: Optional[float] = None
        self._last_bandit_q_range: Optional[float] = None
        self._last_bandit_greedy_sigma: Optional[float] = None

        # Time-to-threshold (first step where mse <= threshold)
        self._bandit_tt_mse: Optional[int] = None

        # Portal diagnostics
        self._eval_auc_portal_taken = 0.0
        self._last_eval_portal_taken: Optional[float] = None

        # Time-to-threshold for portal adoption (e.g., portal_taken_rate >= 0.95 for K consecutive evals)
        self._portal_tt_step: Optional[int] = None
        self._portal_tt_streak: int = 0

        # Stability: number of flips in route choice across eval checkpoints
        self._portal_flip_count: int = 0
        self._last_portal_binary: Optional[int] = None


    # Terminal success helpers
    def _terminal_kind(self, obs) -> Optional[str]:
        """
        Returns:
          "goal" if terminal obs corresponds to goal state,
          "hole" if terminal obs corresponds to hole state,
          None if unknown or not a goal/hole env.
        """
        s = int(np.asarray(obs).reshape(()))
        base = self.env.unwrapped

        # TwoChains-style
        if hasattr(base, "G") and hasattr(base, "H"):
            if s == int(base.G):
                return "goal"
            if s == int(base.H):
                return "hole"
            return None

        # FrozenLake-style (desc is a grid of bytes)
        if hasattr(base, "desc") and hasattr(base, "ncol"):
            desc = np.asarray(base.desc)
            ncol = int(base.ncol)
            r, c = divmod(s, ncol)
            tile = desc[r, c]
            if tile == b"G" or tile == "G":
                return "goal"
            if tile == b"H" or tile == "H":
                return "hole"
            return None

        return None

    def _is_goal(self, obs) -> bool:
        return self._terminal_kind(obs) == "goal"

    # Obs / action
    def _encode_obs(self, obs):
        x = self.obs_adapter(obs)
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        if self.model_type == "mlp":
            x = x.view(x.size(0), -1)
        return x

    @torch.no_grad()
    def _select_action(self, obs, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(self.env.action_space.sample())
        q = self.q_net(self._encode_obs(obs))
        return int(q.argmax(dim=1).item())

    # Replay push / sample
    def _push_transition(self, o, a, r, no, terminated, truncated):
        self.replay.add(o, a, r, no, terminated, truncated)

        if self.env_model is not None:
            s_arr = np.asarray(o)
            s_next_arr = np.asarray(no)

            if s_arr.ndim != 0 or s_next_arr.ndim != 0:
                raise ValueError(
                    f"TabularDynamicsModel expects scalar discrete observations, not {s_arr.shape} and {s_next_arr.shape}."
                )

            s = int(s_arr.reshape(()))
            s_next = int(s_next_arr.reshape(()))

            self.env_model.observe(
                s=s,
                a=int(a),
                r=float(r),
                s_next=s_next,
                terminated=terminated,
                truncated=truncated,
            )

    def _grouping_metrics(self) -> Dict[str, float]:
        """
        Logs whether latent grouping is active and whether buckets are usable.
        """
        out: Dict[str, float] = {}

        if not self.is_per:
            return out

        # Is grouping enabled?
        grouping = getattr(self.cfg.agents.replay, "grouping", None)
        enabled = bool(getattr(grouping, "enabled", False)) if grouping is not None else False
        out["grouping/enabled"] = float(enabled)

        # If PER replay exposes stats
        if hasattr(self.replay, "grouping_stats"):
            try:
                out.update(self.replay.grouping_stats(n_actions=self.n_actions))
            except Exception as e:
                out["grouping/stats_error"] = 1.0
                out["grouping/stats_error_type"] = float(1.0)  # keep numeric; error text goes to stdout
                print(f"[grouping] failed to compute grouping_stats: {type(e).__name__}: {e}")

        # Keyer stats (cache hit rate, etc.)
        keyer = getattr(self.replay, "keyer", None)
        if keyer is not None and hasattr(keyer, "stats"):
            ks = keyer.stats()
            for k, v in ks.items():
                out[f"grouping/keyer_{k}"] = float(v)

        return out

    def _sample_batch(self) -> Dict[str, Any]:
        batch = self.replay.sample(self.cfg.agents.replay.batch_size)

        obs_space = self.env.observation_space  # <-- IMPORTANT: define once for all branches

        mit_cfg = self.mit_cfg
        use_mitigation = (self.is_per and mit_cfg is not None and bool(mit_cfg.enabled))
        method = str(getattr(mit_cfg, "method", "none")).lower() if use_mitigation else "none"

        used_indices = np.array(batch["indices"], copy=True) if batch.get("indices", None) is not None else None

        use_next_obs = True
        groups_used: Optional[List[List[int]]] = None
        group_sizes_used = None

        mit_sample_replace_frac = None
        mit_sample_self_frac = None
        mit_sample_group_size_mean = None
        mit_avg_valid_group_frac = None

        # ---------- helper: robust + fast batch conversion ----------
        def _adapt_obs_batch_np(obs_batch_np: Any) -> np.ndarray:
            """
            Returns float32 numpy batch in the format the network expects:
            - Discrete envs: uses obs_adapter (keeps legacy behavior perfectly)
            - Image envs with fixed HWC: vectorized HWC->CHW
            - Fallback: obs_adapter per item
            """
            x = np.asarray(obs_batch_np)

            # Discrete scalar observations (FrozenLake, bandits, etc.)
            if hasattr(obs_space, "n"):
                # preserve EXACT legacy encoding
                out = np.stack([self.obs_adapter(o) for o in x]).astype(np.float32, copy=False)
                return np.ascontiguousarray(out, dtype=np.float32)

            # Fixed-shape image-like observations (MinAtar: HWC)
            expected_hwc = tuple(obs_space.shape)  # (H,W,C) typically
            if x.ndim == 4 and tuple(x.shape[1:]) == expected_hwc:
                x = x.astype(np.float32, copy=False)
                x = np.transpose(x, (0, 3, 1, 2))  # BCHW
                return np.ascontiguousarray(x, dtype=np.float32)

            # Already BCHW (rare, but safe)
            if x.ndim == 4 and len(expected_hwc) == 3:
                H, W, C = expected_hwc
                if x.shape[1] == C and x.shape[2] == H and x.shape[3] == W:
                    return np.ascontiguousarray(x.astype(np.float32, copy=False))

            # Vector obs (B,D)
            if x.ndim == 2:
                return np.ascontiguousarray(x.astype(np.float32, copy=False))

            # Fallback
            out = np.stack([self.obs_adapter(o) for o in x]).astype(np.float32, copy=False)
            return np.ascontiguousarray(out, dtype=np.float32)

        # Mitigation logic
        if use_mitigation:
            include_self = bool(getattr(mit_cfg, "include_self", True))
            min_group = int(getattr(mit_cfg, "min_group", 1))

            if method == "sample":
                repl_idx: List[Optional[int]] = []
                valid_group_sizes = []
                self_picks = 0
                valid_draws = 0

                for idx in batch["indices"]:
                    idx = int(idx)
                    key = self.replay.idx_to_key[idx]
                    if key is None:
                        raise RuntimeError(f"SAMPLE invariant violated: idx_to_key[{idx}] is None")

                    lst = self.replay.by_sa.get(key, [])
                    if len(lst) < 1:
                        raise RuntimeError(f"SAMPLE invariant violated: by_sa[{key}] is empty for idx={idx}")

                    if (not include_self) and len(lst) <= 1:
                        raise RuntimeError(
                            "SAMPLE with include_self=False and singleton groups creates mixture behavior. Use include_self=True."
                        )

                    valid_draws += 1
                    valid_group_sizes.append(len(lst))

                    if include_self:
                        alt = int(random.choice(lst))
                        if alt == idx:
                            self_picks += 1
                    else:
                        alt = idx
                        while alt == idx:
                            alt = int(random.choice(lst))

                    repl_idx.append(alt)

                chosen = [i for i in repl_idx if i is not None]
                if chosen:
                    fetched = self.replay.fetch(chosen)
                    row_of = {int(i): k for k, i in enumerate(fetched["indices"])}
                    for bi, alt_idx in enumerate(repl_idx):
                        if alt_idx is None:
                            continue
                        k = row_of[int(alt_idx)]
                        batch["next_obs"][bi] = fetched["next_obs"][k]
                        batch["rewards"][bi] = fetched["rewards"][k]
                        batch["terminated"][bi] = fetched["terminated"][k]
                        batch["truncated"][bi] = fetched["truncated"][k]
                        used_indices[bi] = int(alt_idx)

                if used_indices is not None and batch.get("indices", None) is not None:
                    mit_sample_replace_frac = float(np.mean(used_indices != batch["indices"]))
                if valid_draws > 0:
                    mit_sample_self_frac = float(self_picks / float(valid_draws)) if include_self else 0.0
                    mit_sample_group_size_mean = float(np.mean(valid_group_sizes)) if valid_group_sizes else 0.0
                else:
                    mit_sample_self_frac = 0.0
                    mit_sample_group_size_mean = 0.0

            elif method == "avg":
                use_next_obs = False
                max_group = int(getattr(mit_cfg, "max_group", 0))
                gamma = float(self.cfg.agents.gamma)

                groups = self.replay.sibling_groups(
                    batch["indices"],
                    include_self=include_self,
                    min_group=min_group,
                    max_group=max_group,
                )

                for idx_i, g in zip(batch["indices"], groups):
                    if g is None or len(g) == 0:
                        raise RuntimeError(f"AVG invariant violated: empty sibling group for sampled idx={int(idx_i)}")

                groups_used = [[int(x) for x in g] for g in groups]
                group_sizes_used = np.array([len(g) for g in groups_used], dtype=np.int64)
                mit_avg_valid_group_frac = 1.0

                all_unique = np.unique(np.concatenate([np.asarray(g, dtype=np.int64) for g in groups_used]))
                fetched = self.replay.fetch(all_unique)

                rewards_u = fetched["rewards"].astype(np.float32)
                term_u = fetched["terminated"].astype(np.float32)
                trunc_u = fetched["truncated"].astype(np.float32)
                if self.cfg.agents.handle_time_limit_as_terminal:
                    done_u = np.maximum(term_u, trunc_u)
                else:
                    done_u = term_u

                next_obs_u = fetched["next_obs"]

                x_next_np = _adapt_obs_batch_np(next_obs_u)
                x_next_u = torch.from_numpy(x_next_np).to(self.device)

                if self.model_type == "mlp":
                    x_next_u = x_next_u.view(x_next_u.size(0), -1)

                with torch.no_grad():
                    q_online = self.q_net(x_next_u)
                    a_star = q_online.argmax(dim=1)
                    q_tgt = self.target_q_net(x_next_u)
                    v_u = q_tgt.gather(1, a_star.unsqueeze(1)).squeeze(1).detach().cpu().numpy()

                targets_u = rewards_u + (1.0 - done_u) * gamma * v_u
                pos = {int(idx): k for k, idx in enumerate(fetched["indices"])}

                B = len(batch["indices"])
                target_agg = np.empty((B,), dtype=np.float32)
                var_list = []
                for i, g in enumerate(groups_used):
                    t = np.asarray([targets_u[pos[int(j)]] for j in g], dtype=np.float32)
                    target_agg[i] = float(t.mean())
                    var_list.append(float(np.var(t)))

                batch["target_agg"] = target_agg
                batch["mit_avg_groups_used"] = groups_used
                batch["mit_unique_sibling_indices"] = int(all_unique.size)
                batch["mit_total_sibling_refs"] = int(sum(len(g) for g in groups_used))
                batch["mit_target_var_mean"] = float(np.mean(var_list)) if var_list else 0.0
                batch["mit_target_var_max"] = float(np.max(var_list)) if var_list else 0.0

            elif method == "model":
                for bi, (s_raw, a_raw) in enumerate(zip(batch["obs"], batch["actions"])):
                    s = int(np.asarray(s_raw).reshape(()))
                    a = int(a_raw)

                    default = (
                        batch["next_obs"][bi],
                        batch["rewards"][bi],
                        batch["terminated"][bi],
                        batch["truncated"][bi],
                    )

                    s_next, r_new, term_new, trunc_new = self.env_model.sample(s=s, a=a, default=default)
                    batch["next_obs"][bi] = s_next
                    batch["rewards"][bi] = r_new
                    batch["terminated"][bi] = term_new
                    batch["truncated"][bi] = trunc_new

        # Build tensors (safe for ALL envs)
        obs_np = _adapt_obs_batch_np(batch["obs"])
        obs_t = torch.from_numpy(obs_np).to(self.device)

        if use_next_obs:
            next_obs_np = _adapt_obs_batch_np(batch["next_obs"])
            next_obs_t = torch.from_numpy(next_obs_np).to(self.device)
        else:
            next_obs_t = torch.empty(0, device=self.device)

        if self.model_type == "mlp":
            obs_t = obs_t.view(obs_t.size(0), -1)
            if use_next_obs:
                next_obs_t = next_obs_t.view(next_obs_t.size(0), -1)

        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32, device=self.device)
        truncated = torch.as_tensor(batch["truncated"], dtype=torch.float32, device=self.device)

        # -------------------------
        # IS weights
        # -------------------------
        weights_np = batch.get("weights", None)
        weights_raw = None
        is_group_weights = False
        group_stats = None

        if weights_np is None:
            weights = torch.ones((obs_t.shape[0],), dtype=torch.float32, device=self.device)
        else:
            use_group_is = (
                self.is_per and use_mitigation and method in ("sample", "avg", "model")
                and hasattr(self.replay, "compute_group_is_weights")
            )
            if use_group_is:
                gw_raw, n_g, s_g, S, n = self.replay.compute_group_is_weights(batch["indices"], normalize=False)
                gw_raw = np.asarray(gw_raw, dtype=np.float32)
                gw = gw_raw / (float(np.max(gw_raw)) + 1e-12)

                weights_raw = torch.as_tensor(gw_raw, dtype=torch.float32, device=self.device).view(-1)
                weights = torch.as_tensor(gw, dtype=torch.float32, device=self.device).view(-1)

                is_group_weights = True
                group_stats = dict(n_g=n_g, s_g=s_g, S=S, n=n)
            else:
                weights = torch.as_tensor(weights_np, dtype=torch.float32, device=self.device).view(-1)

        indices = batch.get("indices", None)

        out: Dict[str, Any] = dict(
            obs=obs_t,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs_t,
            terminated=terminated,
            truncated=truncated,
            weights=weights,
            indices=indices,
            used_indices=used_indices,
            is_group_weights=is_group_weights,
        )

        if weights_raw is not None:
            out["weights_raw"] = weights_raw
        if group_stats is not None:
            out["_group_stats"] = group_stats

        if group_sizes_used is not None:
            out["group_sizes_used"] = torch.as_tensor(group_sizes_used, dtype=torch.float32, device=self.device)
        if "target_agg" in batch:
            out["target_agg"] = torch.as_tensor(batch["target_agg"], dtype=torch.float32, device=self.device)
        if "mit_avg_groups_used" in batch:
            out["mit_avg_groups_used"] = batch["mit_avg_groups_used"]

        if "mit_unique_sibling_indices" in batch:
            out["mit_unique_sibling_indices"] = int(batch["mit_unique_sibling_indices"])
            out["mit_total_sibling_refs"] = int(batch["mit_total_sibling_refs"])
            out["mit_target_var_mean"] = float(batch.get("mit_target_var_mean", 0.0))
            out["mit_target_var_max"] = float(batch.get("mit_target_var_max", 0.0))

        if mit_avg_valid_group_frac is not None:
            out["mit/avg_valid_group_frac"] = float(mit_avg_valid_group_frac)
        if mit_sample_replace_frac is not None:
            out["mit/sample_replace_frac"] = float(mit_sample_replace_frac)
        if mit_sample_self_frac is not None:
            out["mit/sample_self_frac"] = float(mit_sample_self_frac)
        if mit_sample_group_size_mean is not None:
            out["mit/sample_group_size_mean"] = float(mit_sample_group_size_mean)

        # Replay action distribution
        act_np = actions.detach().cpu().numpy()
        binc = np.bincount(act_np, minlength=self.n_actions)
        self.sample_counts[:self.n_actions] += binc

        # Within-group distortion diagnostics
        if self.is_per and indices is not None and used_indices is not None and hasattr(self.replay, "within_group_used_ratio"):
            sampled_idx = np.asarray(indices, dtype=np.int64)
            used_idx = np.asarray(used_indices, dtype=np.int64)
            ratio, pper, punif = self.replay.within_group_used_ratio(sampled_idx, used_idx)

            out["mit/within_group_ratio_mean"] = float(np.mean(ratio))
            out["mit/within_group_ratio_std"] = float(np.std(ratio))
            out["mit/within_group_pper_mean"] = float(np.mean(pper))
            out["mit/within_group_punif_mean"] = float(np.mean(punif))

        return out

    # Bandit diagnostics
    @torch.no_grad()
    @torch.no_grad()
    def _compute_bandit_metrics(self) -> Dict[str, float]:
        env_base = self.env.unwrapped
        if not hasattr(env_base, "true_means"):
            return {}

        true_means = np.asarray(env_base.true_means, dtype=np.float32)
        q = self.q_net(self._encode_obs(0)).detach().cpu().numpy()[0]

        n = min(len(true_means), len(q))
        true_means = true_means[:n]
        q = q[:n]

        err = q - true_means
        mse = float(np.mean(err ** 2))
        mae = float(np.mean(np.abs(err)))

        q_range = float(np.max(q) - np.min(q))
        q_std = float(np.std(q))

        greedy_arm = int(np.argmax(q))
        greedy_sigma = None

        # Optional: infer sigma for ConalBanditEnv-like envs
        if hasattr(env_base, "sigma_max") and hasattr(env_base, "sigma_min") and hasattr(env_base, "n_arms"):
            n_arms = int(env_base.n_arms)
            if n_arms > 1:
                frac = float(greedy_arm) / float(n_arms - 1)
            else:
                frac = 0.0
            greedy_sigma = float(frac * float(env_base.sigma_max) + float(env_base.sigma_min))
        else:
            greedy_sigma = float("nan")

        metrics: Dict[str, float] = {
            "bandit/mse_q_true": mse,
            "bandit/mae_q_true": mae,
            "bandit/q_range": q_range,
            "bandit/q_std": q_std,
            "bandit/greedy_arm": float(greedy_arm),
            "bandit/greedy_arm_sigma": float(greedy_sigma),
        }

        # Keep per-arm diagnostics (fine for 5 arms)
        for i in range(n):
            metrics[f"bandit/q_arm_{i}"] = float(q[i])
            metrics[f"bandit/true_mean_arm_{i}"] = float(true_means[i])
            metrics[f"bandit/err_arm_{i}"] = float(err[i])

        return metrics

    # Target net updates
    def _maybe_update_target(self):
        if self.global_step > 0 and (self.global_step % self.cfg.agents.target_update.interval == 0):
            tau = self.cfg.agents.target_update.tau
            if tau == 1.0:
                self.algo.hard_update()
            else:
                self.algo.soft_update(tau)
            self.target_updates += 1
            log_metrics({"train/target_updates": self.target_updates}, step=self.global_step)

    # Evaluation
    def evaluate(self, episodes: int = 10) -> Dict[str, float]:
        returns: List[float] = []
        successes: List[int] = []
        lengths: List[int] = []

        # Episode-level portal
        portal_taken_eps: List[int] = []

        # Visit-level portal usage (checks state==portal_from and action==portal_action)
        portal_visits = 0
        portal_taken_visits = 0

        # Identify portal env structure (PortalBridgeGrid)
        base_eval = self.eval_env.unwrapped
        is_portal_env = (
            hasattr(base_eval, "portal_from")
            and hasattr(base_eval, "portal_action")
            and hasattr(base_eval, "ncol")
        )
        if is_portal_env:
            portal_from_rc = tuple(base_eval.portal_from)
            portal_action = int(base_eval.portal_action)
            ncol = int(base_eval.ncol)
        else:
            portal_from_rc = None
            portal_action = None
            ncol = None

        for _ in range(episodes):
            obs, _info = self.eval_env.reset()
            done = False
            ep_ret = 0.0
            ep_len = 0

            ep_took_portal = False

            while not done:
                # If portal env, count portal state visits BEFORE choosing action
                if is_portal_env:
                    s = int(np.asarray(obs).reshape(()))
                    r, c = divmod(s, ncol)
                    at_portal = ((r, c) == portal_from_rc)
                else:
                    at_portal = False

                a = self._select_action(obs, epsilon=0.0)

                # Visit-level portal use: (at portal state) AND (take portal action)
                if at_portal:
                    portal_visits += 1
                    if int(a) == portal_action:
                        portal_taken_visits += 1

                next_obs, r, terminated, truncated, info = self.eval_env.step(a)
                done = terminated or truncated

                ep_ret += float(r)
                ep_len += 1

                # Episode-level portal flag from env info (preferred)
                if isinstance(info, dict) and ("took_portal" in info):
                    ep_took_portal = ep_took_portal or bool(info["took_portal"])

                obs = next_obs

            # Episode end
            returns.append(ep_ret)
            lengths.append(ep_len)

            # Standard success: goal vs hole if detectable, else ep_ret>0 fallback
            kind = self._terminal_kind(obs)
            if kind is None:
                successes.append(1 if ep_ret > 0 else 0)
            else:
                successes.append(1 if kind == "goal" else 0)

            portal_taken_eps.append(1 if ep_took_portal else 0)

        ret = np.asarray(returns, dtype=np.float32)
        succ = np.asarray(successes, dtype=np.float32)
        pt = np.asarray(portal_taken_eps, dtype=np.float32) if portal_taken_eps else None

        success_rate = float(np.mean(succ)) if succ.size > 0 else 0.0
        return_mean = float(np.mean(ret)) if ret.size > 0 else 0.0
        return_std = float(np.std(ret)) if ret.size > 0 else 0.0
        ep_len_mean = float(np.mean(lengths)) if len(lengths) > 0 else 0.0

        q10, q50, q90 = [float(x) for x in np.quantile(ret, [0.10, 0.50, 0.90])] if ret.size > 0 else (0.0, 0.0, 0.0)
        success_any = float(np.max(succ)) if succ.size > 0 else 0.0

        # time-to-first eval goal
        if self.first_eval_goal_step is None and success_any > 0.0:
            self.first_eval_goal_step = int(self.global_step)

        # --- AUC updates (trapezoid rule in env steps) ---
        if self._last_eval_step is not None and self._last_eval_success is not None and self._last_eval_return is not None:
            dt = float(self.global_step - self._last_eval_step)
            if dt > 0:
                self._eval_auc_success += 0.5 * (self._last_eval_success + success_rate) * dt
                self._eval_auc_return += 0.5 * (self._last_eval_return + return_mean) * dt

                # Portal-taken AUC (episode-level rate, in [0,1])
                if pt is not None:
                    portal_taken_rate = float(np.mean(pt)) if pt.size > 0 else 0.0
                    if self._last_eval_portal_taken is not None:
                        self._eval_auc_portal_taken += 0.5 * (self._last_eval_portal_taken + portal_taken_rate) * dt

        # Update last eval point fields
        self._last_eval_step = int(self.global_step)
        self._last_eval_success = float(success_rate)
        self._last_eval_return = float(return_mean)

        # Portal taken rate (episode-level)
        portal_taken_rate = float(np.mean(pt)) if pt is not None and pt.size > 0 else 0.0
        self._last_eval_portal_taken = float(portal_taken_rate)

        # Portal use rate (visit-level)
        portal_use_rate = float(portal_taken_visits) / float(portal_visits + 1e-12) if is_portal_env else float("nan")

        # --- Portal TT + flips ---
        # Configure threshold and consecutive requirement (optional config knobs)
        log_cfg = getattr(self.cfg.agents, "logging", None)
        thr = float(getattr(log_cfg, "portal_tt_threshold", 0.95)) if log_cfg is not None else 0.95
        K = int(getattr(log_cfg, "portal_tt_consecutive", 2)) if log_cfg is not None else 2

        # Binary route decision at checkpoint (use portal_taken_rate, usually near 0 or 1 for greedy eval)
        portal_binary = 1 if portal_taken_rate >= 0.5 else 0
        if self._last_portal_binary is not None and portal_binary != self._last_portal_binary:
            self._portal_flip_count += 1
        self._last_portal_binary = portal_binary

        # streak toward TT
        if portal_taken_rate >= thr:
            self._portal_tt_streak += 1
        else:
            self._portal_tt_streak = 0

        if self._portal_tt_step is None and self._portal_tt_streak >= K:
            self._portal_tt_step = int(self.global_step)

        # Interpret AUC as integral over steps; dividing by steps gives mean-over-training.
        denom = float(max(1, self.global_step))
        auc_success_avg = float(self._eval_auc_success / denom)
        auc_return_avg = float(self._eval_auc_return / denom)
        portal_auc_taken_avg = float(self._eval_auc_portal_taken / denom) if is_portal_env else float("nan")

        metrics: Dict[str, float] = {
            "eval/return_mean": return_mean,
            "eval/return_std": return_std,
            "eval/return_q10": float(q10),
            "eval/return_q50": float(q50),
            "eval/return_q90": float(q90),
            "eval/success_rate": success_rate,
            "eval/success_any": float(success_any),
            "eval/episode_length": ep_len_mean,
            "eval/auc_success": float(self._eval_auc_success),
            "eval/auc_return": float(self._eval_auc_return),

            # Key “table-friendly” scalars:
            "eval/auc_success_avg": auc_success_avg,   # in [0,1]
            "eval/auc_return_avg": auc_return_avg,     # average return over training
        }

        if is_portal_env:
            metrics.update({
                "eval/portal_taken_rate": float(portal_taken_rate),   # episode-level
                "eval/portal_use_rate": float(portal_use_rate),       # visit-level
                "eval/portal_auc_taken": float(self._eval_auc_portal_taken),
                "eval/portal_auc_taken_avg": float(portal_auc_taken_avg),  # in [0,1]

                # Time-to-threshold + stability
                "eval/portal_flip_count": float(self._portal_flip_count),
                "eval/portal_tt_threshold": float(thr),
                "eval/portal_tt_consecutive": float(K),
            })
            if self._portal_tt_step is not None:
                metrics["eval/portal_tt_step"] = float(self._portal_tt_step)

        if self.first_eval_goal_step is not None:
            metrics["eval/first_goal_step"] = float(self.first_eval_goal_step)

        log_metrics(metrics, step=self.global_step)
        self.eval_logs.append({"step": int(self.global_step), **metrics})
        return metrics

    # Training
    def train(self):
        o, _ = self.env.reset()
        ep = 0
        ep_stats = EpisodeStats()

        while self.global_step < self.cfg.train.total_steps:
            epsilon = self.eps_sched(self.global_step)
            a = self._select_action(o, epsilon)
            self.action_counts[a] += 1

            no, r, terminated, truncated, _ = self.env.step(a)
            d = terminated or truncated

            self._push_transition(o, a, r, no, terminated, truncated)
            ep_stats.reward_sum += float(r)
            ep_stats.length += 1

            o = no
            self.global_step += 1

            # Learn
            if (len(self.replay) >= self.cfg.agents.learning_starts) and (self.global_step % self.cfg.agents.train_freq == 0):
                for _ in range(self.cfg.agents.gradient_steps):
                    batch = self._sample_batch()
                    _loss, logs = self.algo.compute_loss(batch, self.optimizer)

                    # PER priority updates
                    if self.is_per and batch.get("indices", None) is not None and "td_errors" in logs:
                        td_abs = logs["td_errors"].detach().abs().cpu().numpy().astype(np.float32)
                        eps_prio = float(self.cfg.agents.replay.eps)
                        prios = td_abs + eps_prio

                        mit_cfg = self.mit_cfg
                        method = str(getattr(mit_cfg, "method", "none")).lower() if mit_cfg else "none"
                        update_all = bool(getattr(mit_cfg, "update_all_siblings", False)) if mit_cfg else False

                        upd_indices = batch.get("used_indices", batch["indices"])
                        upd_indices = np.asarray(upd_indices, dtype=np.int64)

                        if update_all and method == "avg":
                            groups_used = batch.get("mit_avg_groups_used", None)
                            if groups_used is None:
                                self.replay.update_priorities(upd_indices, prios)
                            else:
                                prio_map: Dict[int, float] = {}
                                for i, g in enumerate(groups_used):
                                    pval = float(prios[i])
                                    for j in g:
                                        jj = int(j)
                                        prev = prio_map.get(jj, -1.0)
                                        if pval > prev:
                                            prio_map[jj] = pval
                                idx_list = list(prio_map.keys())
                                prio_list = [prio_map[k] for k in idx_list]
                                self.replay.update_priorities(idx_list, prio_list)
                        else:
                            self.replay.update_priorities(upd_indices, prios)

                    # Logging (only here!)
                    if self.global_step % self.cfg.agents.log_interval_steps == 0:
                        metrics: Dict[str, Any] = {
                            "train/loss": float(logs["loss"]),
                            "train/td_error_mean": float(logs["td_error_mean"]),
                            "train/q_mean": float(logs["q_mean"]),
                            "train/grad_norm": float(logs.get("grad_norm", 0.0)),
                            "train/epsilon": float(epsilon),
                            "buffer/size": int(len(self.replay)),
                            "optim/lr": float(self.optimizer.param_groups[0]["lr"]),
                            "env/step": int(self.global_step),
                        }

                        if "td_errors" in logs:
                            td = logs["td_errors"].detach().cpu().numpy()
                            metrics["train/td_error_std"] = float(np.std(td))
                            metrics["train/td_error_max_abs"] = float(np.max(np.abs(td)))
                            if wandb.run is not None:
                                metrics["train/td_error_hist"] = wandb.Histogram(td)

                        # IS weight diagnostics
                        w = batch["weights"].detach().cpu().numpy()
                        metrics["is/is_group_weights"] = float(batch.get("is_group_weights", False))
                        metrics["is/weights_mean"] = float(np.mean(w))
                        metrics["is/weights_std"] = float(np.std(w))
                        metrics["is/weights_max"] = float(np.max(w))

                        num = float(np.sum(w))
                        den = float(np.sum(w * w))
                        metrics["is/ess"] = float((num * num) / (den + 1e-12))

                        if "weights_raw" in batch:
                            wraw = batch["weights_raw"].detach().cpu().numpy()
                            metrics["is/weights_raw_mean"] = float(np.mean(wraw))
                            metrics["is/weights_raw_std"] = float(np.std(wraw))
                            metrics["is/weights_raw_max"] = float(np.max(wraw))

                        gs = batch.get("_group_stats", None)
                        if gs is not None:
                            n_g = np.asarray(gs["n_g"])
                            s_g = np.asarray(gs["s_g"])
                            S = float(gs["S"])
                            n = float(gs["n"])

                            metrics["is/group_n_g_mean"] = float(np.mean(n_g))
                            metrics["is/group_n_g_max"] = float(np.max(n_g))
                            metrics["is/group_s_g_mean"] = float(np.mean(s_g))
                            metrics["is/group_s_g_min"] = float(np.min(s_g))

                            base = (n_g * S) / (n * (s_g + 1e-12))
                            metrics["is/group_base_mean"] = float(np.mean(base))
                            metrics["is/group_base_max"] = float(np.max(base))

                        # Mitigation diagnostics already attached in batch
                        for k in [
                            "mit/within_group_ratio_mean",
                            "mit/within_group_ratio_std",
                            "mit/within_group_pper_mean",
                            "mit/within_group_punif_mean",
                        ]:
                            if k in batch:
                                metrics[k] = float(batch[k])

                        if "mit/avg_valid_group_frac" in batch:
                            metrics["mit/avg_valid_group_frac"] = float(batch["mit/avg_valid_group_frac"])

                        # Policy/replay action fractions (guarded)
                        if self.n_actions <= self._max_action_frac_logs:
                            total_actions = int(self.action_counts.sum())
                            if total_actions > 0:
                                for a_idx in range(self.n_actions):
                                    metrics[f"policy/frac_action_{a_idx}"] = float(self.action_counts[a_idx]) / float(total_actions)

                            total_samples = int(self.sample_counts.sum())
                            if total_samples > 0:
                                for a_idx in range(self.n_actions):
                                    metrics[f"replay/frac_samples_action_{a_idx}"] = float(self.sample_counts[a_idx]) / float(total_samples)

                        # Grouping metrics (ONLY here, and actually logged)
                        metrics.update(self._grouping_metrics())

                        log_metrics(metrics, step=self.global_step)

                        # Store locally (exclude histograms)
                        self.step_logs.append({
                            "step": int(self.global_step),
                            **{
                                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in metrics.items()
                                if not isinstance(v, wandb.Histogram)
                            },
                        })

                        debug = getattr(self.replay, "debug_snapshot", lambda: None)()
                        if debug is not None:
                            prios_dbg = debug.get("debug_priorities", [])
                            if len(prios_dbg) > 0:
                                debug_metrics = {
                                    "mit/debug_group_size": float(debug.get("debug_group_size", 0)),
                                    "mit/debug_prio_mean": float(np.mean(prios_dbg)),
                                    "mit/debug_prio_max": float(np.max(prios_dbg)),
                                }
                                if "debug_group_mass" in debug:
                                    debug_metrics["mit/debug_group_mass"] = float(debug["debug_group_mass"])
                                log_metrics(debug_metrics, step=self.global_step)
                                self.step_logs[-1].update(debug_metrics)

            # Target update
            self._maybe_update_target()

            # Episode end
            if d:
                kind = self._terminal_kind(o)
                if kind is None:
                    ep_stats.success = 1 if ep_stats.reward_sum > 0 else 0
                else:
                    ep_stats.success = 1 if kind == "goal" else 0

                if self.first_train_goal_step is None and ep_stats.success == 1:
                    self.first_train_goal_step = int(self.global_step)
                    log_metrics({"train/first_goal_step": float(self.first_train_goal_step)}, step=self.global_step)

                ep_metrics = {
                    "train/episode_return": float(ep_stats.reward_sum),
                    "train/episode_length": int(ep_stats.length),
                    "train/episode_success": int(ep_stats.success),
                    "train/epsilon": float(epsilon),
                    "buffer/size": int(len(self.replay)),
                }
                log_metrics(ep_metrics, step=self.global_step)

                self.episode_logs.append({
                    "episode": int(ep),
                    "step": int(self.global_step),
                    **ep_metrics,
                })

                ep += 1
                if (ep % self.cfg.train.eval_interval_episodes) == 0:
                    self.evaluate(episodes=self.cfg.agents.eval_episodes)

                o, _ = self.env.reset()
                ep_stats = EpisodeStats()

    @torch.no_grad()
    def compute_q_values_all_states(self) -> np.ndarray:
        obs_space = self.env.observation_space
        n_states = obs_space.n
        q_table = np.zeros((n_states, self.n_actions), dtype=np.float32)

        for s in range(n_states):
            x = self._encode_obs(s)
            q = self.q_net(x).detach().cpu().numpy()[0]
            q_table[s] = q
        return q_table
