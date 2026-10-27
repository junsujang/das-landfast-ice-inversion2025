#!/bin/bash
#SBATCH --job-name=bfdisp_stitch
#SBATCH --output=logs/stitch_out_%j.txt
#SBATCH --error=logs/stitch_err_%j.txt
#SBATCH --mail-user=junsu.jang@whoi.edu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# STEP 4. Stitch the step-3 per-block posteriors into one small stats file to
# copy home. Run after step 3; to chain them:
#
#     jid=$(sbatch --parsable step3_posterior.slurm)
#     sbatch --dependency=afterok:${jid} step4_stitch.slurm
#
# Stitches the run configured in config.sh. Drop ${DIR} and --is-coherence-norm
# to stitch every step-3 run under outputs/bf_inv_disp/posterior/ instead.

set -euo pipefail

REPO=/user/junsu.jang/das-landfast-ice-inversion2025
source ${REPO}/scripts/bf_inv_disp/slurms/config.sh

python ${SCRIPTS}/step4_stitch_bf_disp_series.py ${DIR} \
    --is-coherence-norm ${COH_NORM}
