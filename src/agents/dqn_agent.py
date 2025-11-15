from dataclasses import dataclass
from typing import Any, Dict
import time
import numpy as np
import torch
import torch.optim as optim
import wandb

from src.replay import make_replay
from src.models.factory import build_q_network
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
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.agents.optimizer.lr, weight_decay=self.cfg.agents.optimizer.weight_decay)

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

    def _push_transition(self, o, a, r, no, terminated, truncated):
        # Store raw obs (int for FrozenLake), though buffer handles types
        self.replay.add(o, a, r, no, terminated, truncated)

    def _sample_batch(self) -> Dict[str, Any]:
        batch = self.replay.sample(self.cfg.agents.replay.batch_size)

        mit_cfg = self.mit_cfg
        use_mitigation = (
            self.is_per
            and mit_cfg is not None
            and bool(mit_cfg.enabled)
        )

        groups = None
        group_sizes = None
        
        if use_mitigation:
            method = str(mit_cfg.method).lower()
            include_self = bool(mit_cfg.include_self)
            min_group = int(mit_cfg.min_group)
            max_group = int(mit_cfg.max_group)

            groups = self.replay.sibling_groups(
                batch["indices"],
                include_self=include_self,
                min_group=min_group,
                max_group=max_group,
            )

            if method == "other":
                # Replace (next_obs, r, term, trunc) transition with another sibling (if exists)
                repl_idx = []
                for g, idx in zip(groups, batch["indices"]):
                    if len(g) == 0:
                        repl_idx.append(None)
                    else:
                        repl_idx.append(int(np.random.choice(g)))

                chosen = [i for i in repl_idx if i is not None]
                if len(chosen) > 0:
                    fetched = self.replay.fetch(chosen)
                    if fetched is None or "indices" not in fetched:
                        pass
                    else:
                        row_of = {int(i): k for k, i in enumerate(fetched["indices"])}
                        for bi, alt_idx in enumerate(repl_idx):
                            if alt_idx is None:
                                continue
                            k = row_of[int(alt_idx)]
                            batch["next_obs"][bi]   = fetched["next_obs"][k]
                            batch["rewards"][bi]    = fetched["rewards"][k]
                            batch["terminated"][bi] = fetched["terminated"][k]
                            batch["truncated"][bi]  = fetched["truncated"][k]

            if method == "avg":
                B = len(batch["indices"])
                target_agg = np.empty((B,), dtype=np.float32)
                gamma = float(self.cfg.agents.gamma)

                for i, (g, idx_i) in enumerate(zip(groups, batch["indices"])):
                    # If no siblings, fall back to the sampled index itself
                    if not g:
                        g = [int(idx_i)]

                    fetched = self.replay.fetch(g)

                    rewards_j = fetched["rewards"].astype(np.float32)
                    term_j    = fetched["terminated"].astype(np.float32)
                    trunc_j   = fetched["truncated"].astype(np.float32)

                    if self.cfg.agents.handle_time_limit_as_terminal:
                        done_j = np.maximum(term_j, trunc_j)
                    else:
                        done_j = term_j

                    next_list = [self.obs_adapter(o) for o in fetched["next_obs"]]
                    x_next = torch.as_tensor(
                        np.stack(next_list),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    
                    if self.model_type == "mlp":
                        x_next = x_next.view(x_next.size(0), -1)

                    with torch.no_grad():
                        q_online = self.q_net(x_next)
                        a_star = q_online.argmax(dim=1)
                        q_tgt = self.target_q_net(x_next)
                        v_j = q_tgt.gather(1, a_star.unsqueeze(1)).squeeze(1).cpu().numpy()

                    targets_j = rewards_j + (1.0 - done_j) * gamma * v_j
                    target_agg[i] = float(targets_j.mean())

                batch["target_agg"] = target_agg

            batch["mitigation_groups"] = groups
            group_sizes = np.array([len(g) for g in groups], dtype=np.int64)

        obs = torch.as_tensor(
            np.stack([self.obs_adapter(o) for o in batch["obs"]]),
            dtype=torch.float32, device=self.device
        )
        next_obs = torch.as_tensor(
            np.stack([self.obs_adapter(o) for o in batch["next_obs"]]),
            dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32, device=self.device)
        truncated = torch.as_tensor(batch["truncated"], dtype=torch.float32, device=self.device)
        
        # This is to implement PER later
        weights = batch.get("weights", None)
        if weights is None:
            weights = torch.ones((obs.shape[0],), dtype=torch.float32, device=self.device)
        else:
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        indices = batch.get("indices", None)
        
        out = dict(
            obs=obs, actions=actions, rewards=rewards,
            next_obs=next_obs, terminated=terminated, truncated=truncated,
            weights=weights, indices=indices
        )
        
        # new metrics for the bias mitigation technique
        if groups is not None:
            out["mitigation_groups"] = groups
            out["group_sizes"] = torch.as_tensor(
                group_sizes, dtype=torch.float32, device=self.device
            )

        if "target_agg" in batch:
            out["target_agg"] = torch.as_tensor(
                batch["target_agg"], dtype=torch.float32, device=self.device
            )

        return out

    def _maybe_update_target(self):
        if self.global_step > 0 and (self.global_step % self.cfg.agents.target_update.interval == 0):
            tau = self.cfg.agents.target_update.tau
            if tau == 1.0:
                self.algo.hard_update()
            else:
                self.algo.soft_update(tau)
            self.target_updates += 1
            log_metrics({"train/target_updates": self.target_updates}, step=self.global_step)

    def evaluate(self, episodes: int = 10) -> Dict[str, float]:
        returns = []
        successes = []
        lengths = []
        for _ in range(episodes):
            obs, info = self.eval_env.reset()
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done:
                a = self._select_action(obs, epsilon=0.0)
                next_obs, r, terminated, truncated, _ = self.eval_env.step(a)
                done = terminated or truncated
                ep_ret += r
                ep_len += 1
                obs = next_obs
            returns.append(ep_ret)
            successes.append(1 if ep_ret > 0 else 0)
            lengths.append(ep_len)
        metrics = {
            "eval/return_mean": float(np.mean(returns)),
            "eval/return_std": float(np.std(returns)),
            "eval/success_rate": float(np.mean(successes)),
            "eval/episode_length": float(np.mean(lengths)),
        }
        log_metrics(metrics, step=self.global_step)

        self.eval_logs.append({
            "step": int(self.global_step),
            **metrics,
        })

        return metrics

    def train(self):
        o, _ = self.env.reset()
        ep = 0
        ep_stats = EpisodeStats()

        while self.global_step < self.cfg.train.total_steps:
            epsilon = self.eps_sched(self.global_step)
            a = self._select_action(o, epsilon)
            no, r, terminated, truncated, _ = self.env.step(a)
            d = terminated or truncated

            self._push_transition(o, a, r, no, terminated, truncated)
            ep_stats.reward_sum += r
            ep_stats.length += 1
            if r > 0:
                ep_stats.success = 1

            o = no
            self.global_step += 1

            # Learn
            if (len(self.replay) >= self.cfg.agents.learning_starts) and (self.global_step % self.cfg.agents.train_freq == 0):
                for _ in range(self.cfg.agents.gradient_steps):
                    batch = self._sample_batch()
                    loss, logs = self.algo.compute_loss(batch, self.optimizer)

                    # PER
                    if batch["indices"] is not None and "td_errors" in logs:
                        td = logs["td_errors"].detach().abs().cpu().numpy()
                        eps_prio = self.cfg.agents.replay.eps
                        prios = td + eps_prio

                        mit_cfg = self.mit_cfg
                        method = str(getattr(mit_cfg, "method", "none")).lower() if mit_cfg else "none"
                        groups = batch.get("mitigation_groups", None)

                        use_family = (
                            self.is_per
                            and mit_cfg is not None
                            and getattr(mit_cfg, "enabled", False)
                            and getattr(mit_cfg, "update_all_siblings", False)
                            and method == "avg"
                            and groups is not None
                        )

                        if use_family:
                            idx_list, prio_list = [], []
                            for i, idx in enumerate(batch["indices"]):
                                # always include the sampled index
                                idx_list.append(int(idx))
                                prio_list.append(float(prios[i]))
                                # then its siblings (groups[i] may exclude self)
                                for sib in groups[i]:
                                    idx_list.append(int(sib))
                                    prio_list.append(float(prios[i]))
                            self.replay.update_priorities(idx_list, prio_list)
                        else:
                            self.replay.update_priorities(batch["indices"], prios)

                    # Logging
                    if self.global_step % self.cfg.agents.log_interval_steps == 0:
                        metrics = {
                            "train/loss": float(logs["loss"]),
                            "train/td_error_mean": float(logs["td_error_mean"]),
                            "train/q_mean": float(logs["q_mean"]),
                            "train/grad_norm": float(logs.get("grad_norm", 0.0)),
                            "train/epsilon": float(epsilon),
                            "buffer/size": len(self.replay),
                            "optim/lr": float(self.optimizer.param_groups[0]["lr"]),
                            "env/step": self.global_step,
                        }

                        # Bias mitigation metrics
                        group_sizes = batch.get("group_sizes", None)
                        if group_sizes is not None:
                            gs_np = group_sizes.detach().cpu().numpy()
                            if gs_np.size > 0:
                                metrics.update({
                                    "mit/group_size_mean": float(gs_np.mean()),
                                    "mit/group_size_max": float(gs_np.max()),
                                    "mit/group_nonempty_frac": float((gs_np > 0).mean()),
                                })

                        if "td_errors" in logs:
                            td = logs["td_errors"].detach().cpu().numpy()
                            metrics["train/td_error_std"] = float(np.std(td))
                            metrics["train/td_error_max_abs"] = float(np.max(np.abs(td)))

                            if wandb.run is not None:
                                metrics["train/td_error_hist"] = wandb.Histogram(td)

                        log_metrics(metrics, step=self.global_step)

                        self.step_logs.append({
                            "step": int(self.global_step),
                            **{
                                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in metrics.items()
                                if not isinstance(v, wandb.Histogram)
                            },
                        })

                        # Bias mitigation for averaging
                        debug = getattr(self.replay, "debug_snapshot", lambda: None)()
                        if debug is not None:
                            prios = debug.get("debug_priorities", [])
                            if len(prios) > 0:
                                debug_metrics = {
                                    "mit/debug_group_size": debug["debug_group_size"],
                                    "mit/debug_prio_mean": float(np.mean(prios)),
                                    "mit/debug_prio_max": float(np.max(prios)),
                                }
                                log_metrics(debug_metrics, step=self.global_step)

                                # Last step_logs entry
                                self.step_logs[-1].update(debug_metrics)
                        
            # Target update
            self._maybe_update_target()

            # Episode end
            if d:
                ep_metrics = {
                    "train/episode_return": ep_stats.reward_sum,
                    "train/episode_length": ep_stats.length,
                    "train/episode_success": ep_stats.success,
                    "train/epsilon": float(epsilon),
                    "buffer/size": len(self.replay),
                }
                log_metrics(ep_metrics, step=self.global_step)

                # Store episode-level metrics locally
                self.episode_logs.append({
                    "episode": int(ep),
                    "step": int(self.global_step),
                    **ep_metrics,
                })

                ep += 1
                # Periodic eval
                if (ep % self.cfg.train.eval_interval_episodes) == 0:
                    self.evaluate(episodes=self.cfg.agents.eval_episodes)

                # Reset episode
                o, _ = self.env.reset()
                ep_stats = EpisodeStats()

    @torch.no_grad()
    def compute_q_values_all_states(self) -> np.ndarray:
        """
        For discrete-state envs like FrozenLake:
        returns an array of shape [n_states, n_actions] with Q(s, ·).
        """
        obs_space = self.env.observation_space
        n_states = obs_space.n
        q_table = np.zeros((n_states, self.n_actions), dtype=np.float32)

        for s in range(n_states):
            x = self._encode_obs(s)  # uses obs_adapter internally
            q = self.q_net(x).detach().cpu().numpy()[0]
            q_table[s] = q
        return q_table
