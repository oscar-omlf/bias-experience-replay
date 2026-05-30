"""Audit exact-group AVG/SAMPLE sampling invariants on a tiny terminal MDP.

The toy buffer has one exact (s, a) group with two terminal reward outcomes.
Priorities are deliberately tilted toward high-reward siblings. A correct
exact-group AVG implementation should average uniformly sampled siblings, so
its target mean must equal the empirical group mean, not the PER-anchor mean.
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.replay.per import PrioritizedReplayBuffer
from src.models.tabular_model import TabularDynamicsModel


def main():
    random.seed(0)
    np.random.seed(0)

    rewards = np.asarray([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    replay = PrioritizedReplayBuffer(capacity=16, obs_shape=1, alpha=0.6, beta=0.4, normalize_is_weights=False)
    for reward in rewards:
        replay.add(0, 0, float(reward), 0, True, False)

    # Mimic PER self-selection: high-reward siblings have much larger priority.
    replay.update_priorities(range(len(rewards)), [1, 1, 1, 1, 100, 100, 100, 100])

    leaf = np.asarray([replay.tree.tree[replay.tree.leaf_idx(i)] for i in range(len(rewards))])
    per_anchor_mean = float(np.sum(rewards * leaf) / replay.tree.total_priority())
    empirical_mean = float(np.mean(rewards))

    print(f"empirical_group_target_mean={empirical_mean:.6f}")
    print(f"per_anchor_target_mean={per_anchor_mean:.6f}")

    avg_variances = []
    for k in [1, 2, 4, 8]:
        avg_targets = []
        anchor_hits = []
        for _ in range(50_000):
            # Condition on high-priority anchor 7. If anchor were forced, the
            # K-subset mean would be biased upward.
            group = replay.sibling_groups([7], include_self=True, min_group=1, max_group=k)[0]
            vals = np.asarray([replay.rewards[j] for j in group], dtype=np.float32)
            avg_targets.append(float(np.mean(vals)))
            anchor_hits.append(float(7 in group))
        target_mean = float(np.mean(avg_targets))
        target_var = float(np.var(avg_targets))
        avg_variances.append(target_var)
        print(
            f"avg_k={k} "
            f"target_mean={target_mean:.6f} "
            f"target_var={target_var:.6f} "
            f"anchor_inclusion={np.mean(anchor_hits):.6f}"
        )
        assert abs(target_mean - empirical_mean) < 0.15

    sample_targets = []
    group = replay.by_sa[replay.idx_to_key[7]]
    for _ in range(50_000):
        sample_targets.append(float(replay.rewards[random.choice(group)]))
    sample_target_mean = float(np.mean(sample_targets))
    sample_target_var = float(np.var(sample_targets))
    print(f"sample_target_mean={sample_target_mean:.6f}")
    print(f"sample_target_var={sample_target_var:.6f}")
    assert abs(sample_target_mean - empirical_mean) < 0.15
    assert per_anchor_mean > empirical_mean + 3.0
    assert all(v1 > v2 for v1, v2 in zip(avg_variances, avg_variances[1:]))
    assert abs(avg_variances[0] - sample_target_var) < 0.05

    weights, n_g, s_g, S, n = replay.compute_group_is_weights([0, 7], beta=1.0, normalize=False)
    print(f"group_is_weights={weights.tolist()} n_g={n_g.tolist()} s_g={s_g.tolist()} S={S:.6f} n={n}")
    assert np.allclose(weights, [1.0, 1.0])

    model = TabularDynamicsModel(n_states=3, n_actions=1)
    z1 = (1, 7.0, True, False)
    z2 = (2, 7.0, False, True)
    model.observe(s=0, a=0, s_next=z1[0], r=z1[1], terminated=z1[2], truncated=z1[3])
    model.observe(s=0, a=0, s_next=z2[0], r=z2[1], terminated=z2[2], truncated=z2[3])
    sampled = {model.sample(s=0, a=0) for _ in range(200)}
    print(f"model_joint_outcomes={sorted(sampled)}")
    assert sampled == {z1, z2}

    model.unobserve(s=0, a=0, s_next=z1[0], r=z1[1], terminated=z1[2], truncated=z1[3])
    assert model.count(s=0, a=0) == 1
    assert model.sample(s=0, a=0) == z2
    print("audit_passed=true")


if __name__ == "__main__":
    main()
