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


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
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

    # Close
    env.close()
    eval_env.close()

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()