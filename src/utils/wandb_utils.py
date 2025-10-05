from typing import Any, Dict, Optional
import wandb

def setup_wandb(cfg, config_dict=None):
    mode = cfg.wandb.mode
    if mode == "disabled":
        return None
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=mode,
        group=cfg.wandb.group,
        job_type=cfg.wandb.job_type,
        tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        config=config_dict,
    )
    return run


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    try:
        if getattr(wandb, "run", None) is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass
