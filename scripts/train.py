import os
from dataclasses import asdict

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.seed import set_global_seeds
from src.utils.wandb_utils import setup_wandb
from src.envs.registry import make_env
from src.agents.dqn_agent import DQNAgent


def _select_device(device_str: str):
    import torch
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def run_training(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = _select_device(cfg.device)
    print("Device:", device)

    # Seeding
    set_global_seeds(cfg.seed)

    # W&B
    run = setup_wandb(cfg, config_dict=OmegaConf.to_container(cfg, resolve=True))

    # Env
    env, eval_env, obs_adapter = make_env(cfg.env, seed=cfg.seed)

    # Agent
    agent = DQNAgent(
        cfg=cfg,
        env=env,
        eval_env=eval_env,
        obs_adapter=obs_adapter,
        device=device,
    )

    # Train
    agent.train()

    final_eval = agent.evaluate(episodes=cfg.agents.eval_episodes)

    q_values = None
    try:
        save_q_values = bool(getattr(cfg.train, "save_q_values", True))
        max_states = int(getattr(cfg.train, "max_q_values_states", 2048))
        obs_space = getattr(env, "observation_space", None)

        if save_q_values and obs_space is not None and hasattr(obs_space, "n") and int(obs_space.n) <= max_states:
            q_values = agent.compute_q_values_all_states()
    except Exception as e:
        print(f"[train.py] Skipping q_values export due to: {type(e).__name__}: {e}")
        q_values = None

    # Close
    env.close()
    eval_env.close()

    if run is not None:
        run.finish()

    summary = {
        "final_eval": final_eval,
        "total_steps": agent.global_step,
        "episode_logs": getattr(agent, "episode_logs", []),
        "step_logs": getattr(agent, "step_logs", []),
        "eval_logs": getattr(agent, "eval_logs", []),
        "q_values": q_values,
    }
    return summary


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    _ = run_training(cfg)


if __name__ == "__main__":
    main()