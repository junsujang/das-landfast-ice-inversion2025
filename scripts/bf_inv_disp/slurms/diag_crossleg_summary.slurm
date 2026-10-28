#!/bin/bash
#SBATCH --job-name=bfdisp_xleg_sum
#SBATCH --output=logs/xleg_sum_out_%j.txt
#SBATCH --error=logs/xleg_sum_err_%j.txt
#SBATCH --mail-user=junsu.jang@whoi.edu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# DIAGNOSTIC, phase 2 of diag_crossleg.slurm: the leg-wise and cross-leg shares
# at the theta-window estimate, summarized over every block. The blocks are
# spread over the job's CPUs (--workers defaults to $SLURM_CPUS_PER_TASK); each
# worker holds one block's B at a time, so --mem must cover that many.
#
# Chain it to the phase-1 array so it starts once every shard has finished:
#
#   jid=$(sbatch --parsable diag_crossleg.slurm)
#   sbatch --dependency=afterok:${jid} diag_crossleg_summary.slurm
#
# Re-run it alone for another window (--theta-fix / --theta-half-width) or
# frequency range (--fmin / --fmax); phase 1 does not depend on either.

set -euo pipefail

REPO=/user/junsu.jang/das-landfast-ice-inversion2025
source ${REPO}/scripts/bf_inv_disp/slurms/config.sh

# One BLAS thread per worker process.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python ${SCRIPTS}/diag_crossleg_share.py ${DIR} \
    --is-coherence-norm ${COH_NORM} \
    --summary-only \
    --theta-fix 294 --theta-half-width 5
