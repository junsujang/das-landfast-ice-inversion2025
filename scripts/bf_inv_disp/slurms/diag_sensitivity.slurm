#!/bin/bash
#SBATCH --job-name=bfdisp_sens
#SBATCH --output=logs/sens_out_%A_%a.txt
#SBATCH --error=logs/sens_err_%A_%a.txt
#SBATCH --mail-user=junsu.jang@whoi.edu
#SBATCH --array=0-3
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# DIAGNOSTIC. How the (E, P) estimate of one block moves when nu, h_w, rho_i or
# rho_w is changed one at a time. Builds its own steering vectors, but reads the
# band and baseline ice config from the step-1 cache.
#
# The array index is the parameter: 0 nu, 1 h_w, 2 rho_i, 3 rho_w. Each element
# writes its own sensitivity_<param>.npz/.csv. Already-saved values are skipped,
# so a job that hits the time limit can be resubmitted.
#
# The block defaults to das_20250506_000000_5Hz.h5; add --data-file to change it.

set -euo pipefail

REPO=/user/junsu.jang/das-landfast-ice-inversion2025
source ${REPO}/scripts/bf_inv_disp/slurms/config.sh

python ${SCRIPTS}/diag_sensitivity_ice_params.py ${DIR} \
    --is-coherence-norm ${COH_NORM} \
    --param-index ${SLURM_ARRAY_TASK_ID}
