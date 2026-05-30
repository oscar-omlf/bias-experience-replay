# Correcting Within-Group Bias in Prioritized Replay

This repository contains the implementation used to study within-group
self-selection bias in prioritized experience replay (PER). PER is useful
because it allocates updates unevenly across state-action groups. In stochastic
environments, however, it can also oversample particular realized outcomes
within an exact state-action group.

The code implements three sibling-aware variants that preserve PER's
between-group prioritization while correcting its within-group outcome law:

| Variant | `--agent` value | Trained-on target |
| --- | --- | --- |
| Uniform replay (DDQN) | `dqn` | Uniform replay transition |
| PER | `per` | PER-selected transition |
| SAMPLE | `per_sib_sample` | Uniform sibling from the selected group |
| AVG | `per_sib_avg` | Mean Bellman target over a uniform sibling subset |
| MODEL | `per_model` | Outcome sampled from an empirical tabular model |

For exact discrete environments, siblings share the same exact `(state,
action)` pair. SAMPLE is the central correction. AVG is a lower-variance
alternative: by default it averages a uniformly sampled subset of four
siblings and updates the priorities of the siblings used in the estimate.
MODEL samples a joint outcome `(next_state, reward, terminated, truncated)`
from the finite replay buffer's empirical model. MODEL is available only for
discrete state spaces.

For MinAtar, exact repeated states are too sparse to be useful. The repository
also includes the paper's scaling extension: a frozen VQ-VAE maps observations
to latent groups before applying SAMPLE. This extension is intentionally
separate from the exact-group theory.

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate bias-experience-replay
```

For a pip-only CPU environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_cpu.txt
```

GPU use is optional. The exact-group experiments are inexpensive on CPU.
MinAtar runs are substantially faster on a GPU.

## Repository Layout

```text
config/                 Hydra configurations
scripts/train.py        Single-run training entrypoint
scripts/run_experiments.py
                        Multi-seed runner with local CSV and NumPy exports
scripts/train_vqvae.py  VQ-VAE training and latent-group diagnostics
scripts/analyze_results.py
                        Generic curve and AUC plotting utility
scripts/audit_avg_exact_groups.py
                        Tiny finite-buffer AVG/SAMPLE invariant check
src/                    Agents, replay buffers, models, and environments
```

Weights & Biases logging is disabled by default. To enable it, append
`wandb.mode=online wandb.project=<project> wandb.entity=<entity>` to a command.

## Quick Check

The following audit constructs one exact sibling group with deliberately
tilted priorities. It verifies that SAMPLE and AVG recover the empirical group
mean while PER's anchor mean remains tilted. As the AVG subset size grows, its
variance decreases.

```bash
python -m scripts.audit_avg_exact_groups
```

## Exact-Group Experiments

`scripts.run_experiments` saves the resolved config, local logs, summary JSON,
and discrete-state Q-values under `results/<environment>/<agent>/seed_XX/`.
Pass Hydra overrides after `--override`.

Run the two-action outlier bandit:

```bash
for agent in dqn per per_sib_sample per_sib_avg per_model; do
  python -m scripts.run_experiments \
    --agent "$agent" \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --override env=outlierbandit train.total_steps=100000 \
      agents.learning_starts=1000 agents.gamma=0.0
done
```

The safe arm returns `2`. The rare-tail arm returns `100` with probability
`0.01` and `0` otherwise, so its expectation is `1`. This separates value
estimation from exploration complexity.

Run TwoChains:

```bash
for agent in dqn per per_sib_sample per_sib_avg per_model; do
  python -m scripts.run_experiments \
    --agent "$agent" \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --override env=twochains train.total_steps=200000
done
```

TwoChains turns the bandit mechanism into a control problem. The safe branch
has a delayed deterministic reward. The risky branch has a rare high upside
but lower expected value.

Run the main FrozenLake H100/H300 stress test:

```bash
for agent in dqn per per_sib_sample per_sib_avg per_model; do
  python -m scripts.run_experiments \
    --agent "$agent" \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --override env=frozenlake train.total_steps=500000
done
```

The default FrozenLake preset is an 8-by-8 map with intended-action
probability `0.99`, goal reward `100`, hole reward `-300`, and step reward
`-0.01`. The milder H50/H100 preset used in the controlled comparison is
available as `env=frozenlake_h50`.

For AVG, change the sibling subset size with:

```bash
agents.replay.sa_mitigation.max_group=2
```

Use `max_group=0` only when intentionally averaging the full empirical group.
Its cost grows with replay-buffer group size.

Plot a locally saved learning curve and AUC summary:

```bash
python -m scripts.analyze_results \
  --results-root results \
  --env-key FrozenLake-H100-8x8 \
  --metric eval/return_mean \
  --bin-size 10000 \
  --smooth-window 5
```

## MinAtar Extension

The MinAtar extension uses sticky actions with probability `0.1`. The
mean-preserving-tail setting perturbs positive training rewards only:

```text
reward - M                         with probability p_bad
reward + p_bad * M / (1 - p_bad)   otherwise
```

The default values are `p_bad=0.02` and `M=30`. Evaluation rewards remain
unmodified. This changes tail shape without changing the conditional expected
training reward.

Train a Breakout VQ-VAE:

```bash
python -m scripts.train_vqvae \
  env=minatar \
  env.id=MinAtar/Breakout-v0 \
  vqvae.grid_size='[2,1]' \
  vqvae.codebook_size=16 \
  vqvae.embed_dim=64 \
  outputs.save_path=artifacts/vqvae_breakout_2x1_k16_d64_b025.pt
```

The selected default is a `2 x 1` latent grid with a 16-entry codebook and
64-dimensional embeddings. The VQ-VAE is trained once per game and then kept
fixed during RL training. For Space Invaders, use `vqvae.codebook_size=32`.

Run latent SAMPLE on mean-preserving-tail Breakout:

```bash
python -m scripts.run_experiments \
  --agent per_sib_sample \
  --seeds 0 1 2 3 4 \
  --override env=minatar env.id=MinAtar/Breakout-v0 \
    env.mp_tail.enabled=true \
    agents.model.type=cnn \
    agents.optimizer.lr=6.0e-5 \
    agents.replay.grouping.enabled=true \
    agents.replay.grouping.type=vqvae \
    agents.replay.grouping.ckpt_path=artifacts/vqvae_breakout_2x1_k16_d64_b025.pt \
    train.total_steps=5000000
```

For a clean MinAtar run, set `env.mp_tail.enabled=false`. For DDQN and PER
baselines, use `--agent dqn` or `--agent per` and omit the VQ-VAE grouping
overrides. Learning rates were selected per game and method; the command above
is an example for Breakout rather than a universal setting.
