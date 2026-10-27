#!/bin/bash
#SBATCH --job-name=bfdisp_steer
#SBATCH --output=logs/steer_out_%j.txt
#SBATCH --error=logs/steer_err_%j.txt
#SBATCH --mail-user=junsu.jang@whoi.edu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# STEP 1. Dispersion-curve grid + per-frequency steering cache.
#
# The band and frequency decimation are set HERE ONLY; stage 1 records them in
# steering_meta_<postfix>.json and steps 2-4 read them back from there.
#
# Check the cache size first (prints the footprint, writes nothing):
#   python ${SCRIPTS}/step1_steer_vec_disp.py ${DIR} --fmin 0.10 --fmax 1.00 --f-bin-skip 2 --dry-run
#
# Single task: finished frequency files are skipped, so a job that hits the wall
# clock can be resubmitted unchanged.

set -euo pipefail

REPO=/user/junsu.jang/das-landfast-ice-inversion2025
source ${REPO}/scripts/bf_inv_disp/slurms/config.sh

python ${SCRIPTS}/step1_steer_vec_disp.py ${DIR} \
    --fmin 0.10 --fmax 1.00 --f-bin-skip 2
