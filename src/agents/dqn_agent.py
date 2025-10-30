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

    def _sample_batch(self) -> Dict[str, torch.Tensor]:
        batch = self.replay.sample(self.cfg.agents.replay.batch_size)

        




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
        
        # dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        # This is to implement PER later
        weights = batch.get("weights", None)
        if weights is None:
            weights = torch.ones((obs.shape[0],), dtype=torch.float32, device=self.device)
        else:
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        indices = batch.get("indices", None)
        
        return dict(
            obs=obs, actions=actions, rewards=rewards, 
            next_obs=next_obs, terminated=terminated, truncated=truncated,
            weights=weights, indices=indices
        )

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

                        self.replay.update_priorities(batch["indices"], prios)

                    if self.global_step % self.cfg.agents.log_interval_steps == 0:
                        log_metrics({
                            "train/loss": float(logs["loss"]),
                            "train/td_error_mean": float(logs["td_error_mean"]),
                            "train/q_mean": float(logs["q_mean"]),
                            "train/grad_norm": float(logs.get("grad_norm", 0.0)),
                            "train/epsilon": float(epsilon),
                            "buffer/size": len(self.replay),
                            "optim/lr": float(self.optimizer.param_groups[0]["lr"]),
                            "env/step": self.global_step,
                        }, step=self.global_step)
                        
            # Target update
            self._maybe_update_target()

            # Episode end
            if d:
                log_metrics({
                    "train/episode_return": ep_stats.reward_sum,
                    "train/episode_length": ep_stats.length,
                    "train/episode_success": ep_stats.success,
                    "train/epsilon": float(epsilon),
                    "buffer/size": len(self.replay),
                }, step=self.global_step)

                ep += 1
                # Periodic eval
                if (ep % self.cfg.train.eval_interval_episodes) == 0:
                    self.evaluate(episodes=self.cfg.agents.eval_episodes)

                # Reset episode
                o, _ = self.env.reset()
                ep_stats = EpisodeStats()
