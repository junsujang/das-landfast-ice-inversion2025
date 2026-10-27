#!/bin/bash
#SBATCH --job-name=bfdisp
#SBATCH --output=logs/bf_out_%A_%a.txt
#SBATCH --error=logs/bf_err_%A_%a.txt
#SBATCH --mail-user=junsu.jang@whoi.edu
#SBATCH --array=0-9
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# STEP 2. Beamform every covariance block against the step-1 steering cache.
# Run after step 1 has finished.
#
# The array index is the shard index. Keep the array 0-based and NUM_SHARDS equal
# to its size; both are passed explicitly so a partial rerun (sbatch --array=3,7)
# still uses the original split. --array=0-9%4 runs the same 10 shards 4 at a time.
#
# Memory: ~4x the per-frequency cache file that step 1 --dry-run prints.

set -euo pipefail

NUM_SHARDS=10
REPO=/user/junsu.jang/das-landfast-ice-inversion2025
source ${REPO}/scripts/bf_inv_disp/slurms/config.sh

python ${SCRIPTS}/step2_bf_disp_hourly.py ${DIR} \
    --is-coherence-norm ${COH_NORM} \
    --num-shards ${NUM_SHARDS} --shard-index ${SLURM_ARRAY_TASK_ID:-0}
