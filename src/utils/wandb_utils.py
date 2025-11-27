from typing import Any, Dict, Optional
import wandb

def setup_wandb(cfg, config_dict=None):
    mode = cfg.wandb.mode
    if mode == "disabled":
        return None
    
    env_name = "FrozenLake"
    env_id = cfg.env.id
    map_name = getattr(cfg.env, "map_name", None)
    if map_name:
        env_name = f"{env_id}-{map_name}"
    else:
        env_name = env_id

    replay = cfg.agents.replay.type
    seed = cfg.seed

    mit_cfg = getattr(cfg.agents.replay, "sa_mitigation", None)
    if mit_cfg is not None and bool(getattr(mit_cfg, "enabled", False)):
        mit_method = str(getattr(mit_cfg, "method"))
        mit_label = f"mit-{mit_method}"
    else:
        mit_label = "nomit"

    run_name = f"{env_name}_{replay}_{mit_label}_seed{seed}"

    group = getattr(cfg.wandb, "group", None)
    if not group:
        group = "FrozenLake-8x8"

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=mode,
        name=run_name,
        job_type=cfg.wandb.job_type,
        config=config_dict,
        group=group,
    )
    return run


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    try:
        if getattr(wandb, "run", None) is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass
