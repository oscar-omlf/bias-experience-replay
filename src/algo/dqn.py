from typing import Dict, Tuple
import torch
import torch.nn.functional as F


class DoubleDQN:
    def __init__(self, q_net, target_q_net, gamma: float = 0.99, max_grad_norm: float = 10.0, device: str = "cpu", handle_time_limit_as_terminal: bool = True):
        self.q_net = q_net
        self.target_q_net = target_q_net
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.handle_time_limit_as_terminal = handle_time_limit_as_terminal

    def compute_loss(self, batch: Dict, optimizer) -> Tuple[torch.Tensor, Dict]:
        obs = batch["obs"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        terminated = batch["terminated"]
        truncated = batch["truncated"]
        weights = batch.get("weights", None)

        rewards_agg = batch.get("rewards_agg", None)
        target_agg = batch.get("target_agg", None)

        if weights is None:
            weights = torch.ones_like(rewards)

        # Q(s, a)
        q_values = self.q_net(obs)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if target_agg is not None:
                target = target_agg
            else:
                next_q_online = self.q_net(next_obs)
                next_actions = next_q_online.argmax(dim=1)
                next_q_target = self.target_q_net(next_obs)
                next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
                dones = torch.maximum(terminated, truncated) if self.handle_time_limit_as_terminal else terminated

                r = rewards_agg if rewards_agg is not None else rewards
                target = r + (1.0 - dones) * self.gamma * next_q

        # loss = F.smooth_l1_loss(q_sa, target)
        per_sample_loss = F.smooth_l1_loss(q_sa, target, reduction="none")
        loss = (weights * per_sample_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        
        grad_norm = None
        if self.max_grad_norm and self.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            grad_norm = float(grad_norm.item())
        
        optimizer.step()
        with torch.no_grad():
            td_errors = (q_sa - target)
            td_error_mean = float(td_errors.abs().mean().item())
            q_mean = float(q_values.mean().item())

        logs = {
            "loss": float(loss.item()),
            "td_errors": td_errors.detach(),
            "td_error_mean": td_error_mean,
            "q_mean": q_mean,
            "grad_norm": grad_norm if grad_norm is not None else 0.0,
        }
        return loss, logs

    @torch.no_grad()
    def hard_update(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    @torch.no_grad()
    def soft_update(self, tau: float):
        for target_param, online_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
