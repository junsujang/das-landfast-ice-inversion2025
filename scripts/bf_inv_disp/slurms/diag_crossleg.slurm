#!/bin/bash
#SBATCH --job-name=bfdisp_xleg
#SBATCH --output=logs/xleg_out_%A_%a.txt
#SBATCH --error=logs/xleg_err_%A_%a.txt
#SBATCH --mail-user=junsu.jang@whoi.edu
#SBATCH --array=0
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# DIAGNOSTIC. Leg-wise and cross-leg shares of the beam (Text S8).
# Needs the step-1 cache; reuses step-2 blocks if they exist, otherwise beamforms.
#
# The leg split (M1, channels before the bend) comes from the leg geometry file;
# add --m1 N to override it.
#
# Two phases, like step 3. This array writes the per-block arrays; the summary
# is diag_crossleg_summary.slurm, chained so it starts once every shard is done:
#
#   jid=$(sbatch --parsable diag_crossleg.slurm)
#   sbatch --dependency=afterok:${jid} diag_crossleg_summary.slurm

set -euo pipefail

NUM_SHARDS=10
REPO=/user/junsu.jang/das-landfast-ice-inversion2025
source ${REPO}/scripts/bf_inv_disp/slurms/config.sh

# python ${SCRIPTS}/diag_crossleg_share.py ${DIR} \
#     --is-coherence-norm ${COH_NORM} \
#     --num-shards ${NUM_SHARDS} --shard-index ${SLURM_ARRAY_TASK_ID:-0} \
#     --no-summary

python ${SCRIPTS}/diag_crossleg_share.py ${DIR} \
    --is-coherence-norm ${COH_NORM} --summary-only \
    --theta-fix 294 --theta-half-width 5 \
    --fmin 0.10 --fmax 1.00
