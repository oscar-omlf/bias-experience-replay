#!/bin/bash
#SBATCH --job-name=per_frozenlake_fixed
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=8:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs

module purge
module load 2024
module load Miniconda3/24.7.1-0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ber

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

AGENTS=(
  "dqn"
  "per"
  "per_sib_sample"
  # "per_model"
  # "per_sib_avg"
)

EXTRA_OVERRIDES=(
  ""
  ""
  ""
  # ""
  # "agents.replay.sa_mitigation.max_group=4"
)

AGENT="${AGENTS[$SLURM_ARRAY_TASK_ID]}"
EXTRA="${EXTRA_OVERRIDES[$SLURM_ARRAY_TASK_ID]}"

# --- Common overrides ---
# Adjust env id if your Hydra env key differs
BASE_OVERRIDES=(
  # "env=outlierbandit"
  # "env.success_rate=0.99"
  # "env.reward_schedule=[50.0,-100.0,-0.01]"
  "env=portalbridgegrid"
  "agents.gamma=0.99"
  "agents.learning_starts=5000"
  "agents.replay.batch_size=128"
  "agents.target_update.interval= 1000"
  "agents.replay.capacity=50000"
  "agents.epsilon.decay_steps=500000"
  "train.total_steps=500000"
)

# Seeds for final runs
SEEDS=(0)

echo "Running agent=${AGENT}"
echo "Extra override=${EXTRA}"

# Build command
CMD=(python -m scripts.run_experiments
  --agent "${AGENT}"
  --seeds "${SEEDS[@]}"
  --results-root "results/conalbandits/${SLURM_JOB_ID}"
  --override "${BASE_OVERRIDES[@]}"
)

# Add per-job extra override (only for AVG)
if [[ -n "${EXTRA}" ]]; then
  CMD+=( "${EXTRA}" )
fi

echo "Command: ${CMD[@]}"
"${CMD[@]}"

echo "Done agent=${AGENT} task=${SLURM_ARRAY_TASK_ID}"
