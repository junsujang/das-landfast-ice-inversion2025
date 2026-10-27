#!/bin/bash
#SBATCH --job-name=dasspec
#SBATCH --output=logs/spec_out_%j.txt
#SBATCH --error=logs/spec_err_%j.txt
#SBATCH --mail-user=junsu.jang@whoi.edu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# DAS spectrogram (30-min blocks, every bend-relative channel) of the 5 Hz
# decimated hourly files, for figure 3. One pass over the whole record; it reads
# one hourly file at a time, so memory stays small.
#
# Writes ${DAS_LFI_OUTPUT_DIR}/das_spectrogram/das_spectrogram.npz. Copy that file
# to outputs/das_spectrogram/ of the checkout that draws figure 3.

set -euo pipefail

REPO=/user/junsu.jang/das-landfast-ice-inversion2025

# Read by das_tools.utils.proj_path (same locations as scripts/bf_inv_disp/slurms/config.sh).
export DAS_LFI_HOURLY_DIR=/user/junsu.jang/data/DAS/pisces_2025_Fs5Hz
export DAS_LFI_OUTPUT_DIR=${REPO}/outputs

source ${REPO}/.venv/bin/activate
mkdir -p logs
echo "job ${SLURM_JOB_ID:-none} on $(hostname)"
echo "data    : ${DAS_LFI_HOURLY_DIR}"
echo "results : ${DAS_LFI_OUTPUT_DIR}/das_spectrogram"

python ${REPO}/scripts/das_spectrogram/main_spectrogram.py --min-segments 1
