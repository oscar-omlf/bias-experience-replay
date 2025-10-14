from .q_network import QNetwork
from .q_cnn import CNNQNetwork
import numpy as np


def build_q_network(model_cfg, obs_space, n_actions: int):
    mtype = str(model_cfg.type)
    act = str(model_cfg.activation)

    if mtype == "mlp":
        if hasattr(obs_space, "n"):
            input_dim = int(obs_space.n)
        else:
            input_dim = int(np.prod(obs_space.shape))
        net = QNetwork(
            input_dim,
            n_actions,
            hidden_sizes=tuple(model_cfg.hidden_sizes),
            activation=act
        )
        info = {"type":"mlp", "input_dim": input_dim}
        return net, info

    elif mtype == "cnn":
        # Expect obs shape HWC from env space; use C = shape[-1]
        shape = tuple(obs_space.shape)
        in_ch = int(shape[-1])
        net = CNNQNetwork(
            in_channels=in_ch,
            n_actions=n_actions,
            conv_channels=tuple(model_cfg.conv_channels),
            fc_hidden=int(model_cfg.fc_hidden),
            activation=act
        )
        info = {"type":"cnn", "in_ch": in_ch, "spatial": shape[:2]}
        return net, info

    else:
        raise ValueError(f"Unknown model type: {mtype}")
