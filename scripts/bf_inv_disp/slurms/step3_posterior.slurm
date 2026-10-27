#!/bin/bash
#SBATCH --job-name=bfdisp_post
#SBATCH --output=logs/post_out_%j.txt
#SBATCH --error=logs/post_err_%j.txt
#SBATCH --mail-user=junsu.jang@whoi.edu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# STEP 3. Posterior per block, time series and figures. Run after step 2.
#
# Scores over the whole step-1 band by default; add --fmin/--fmax to score a
# sub-band (each window gets its own output directory, nothing is re-beamformed).
#
# To shard the per-block phase: make this an array, add
# `--num-shards N --shard-index ${SLURM_ARRAY_TASK_ID} --no-plots`, and once all
# shards are done run the same command with `--plots-only` instead.

set -euo pipefail

REPO=/user/junsu.jang/das-landfast-ice-inversion2025
source ${REPO}/scripts/bf_inv_disp/slurms/config.sh

python ${SCRIPTS}/step3_analyze_bf_disp_posterior.py ${DIR} \
    --is-coherence-norm ${COH_NORM}
