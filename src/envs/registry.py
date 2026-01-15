import gymnasium as gym
from typing import Tuple, Callable, Dict

from .frozenlake_env import make_frozenlake
# from .miniatar_env import make_miniatar
from .twochains import make_twochains
from .conalbandits_env import make_conal_bandit
from .noisygridworld_env import make_noisy_gridworld
from .outlierbandit_env import make_outlierbandit
from .portalbridgegrid_env import make_portalbridgegrid


_ENV_MAKERS: Dict[str, Callable] = {
    "FrozenLake": make_frozenlake,
    "TwoChains": make_twochains,
    "ConalBandit": make_conal_bandit,
    "ConalBanditShifted": make_conal_bandit,
    "NoisyGridworld": make_noisy_gridworld,
    "OutlierBandit": make_outlierbandit,
    "PortalBridgeGrid": make_portalbridgegrid,
}

def make_env(env_cfg, seed: int) -> Tuple[gym.Env, gym.Env, Callable]:
    """Factory for training and evaluation envs + observation adapter.
    Returns (env, eval_env, obs_adapter) where obs_adapter(obs) -> np.ndarray.
    """
    env_id = env_cfg.id

    # if env_id.startswith("MinAtar/"):
    #     return make_miniatar(env_cfg, seed)

    base_id = env_id.split("-", 1)[0]
    maker = _ENV_MAKERS.get(base_id)
    if maker is None:
        raise ValueError(f"Unsupported env id: {env_id} (base_id={base_id})")

    return maker(env_cfg, seed)
