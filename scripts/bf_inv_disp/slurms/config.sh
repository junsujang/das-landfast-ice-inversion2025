# Settings shared by every bf_inv_disp job. Sourced by the *.slurm files after
# they set REPO; edit paths and the run configuration here, not in the jobs, so
# all stages always agree.

VENV=${REPO}/.venv
SCRIPTS=${REPO}/scripts/bf_inv_disp

# Read by das_tools.utils.proj_path in every script (see README "Data and outputs").
export DAS_LFI_HOURLY_DIR=${REPO}/data/DAS/pisces_2025_Fs5Hz   # decimated hourly files
export DAS_LFI_OUTPUT_DIR=${REPO}/outputs                                # results in <this>/bf_inv_disp

# Directional weighting of the steering vectors. It names the steering cache, so
# every stage needs it. floor 1.0 / power 0 = omnidirectional ("omni").
DIR="--directional-floor 1.0 --directional-power 0"

# Normalize each covariance to a coherence matrix (stages 2-4 and diagnostics).
COH_NORM=true

source ${VENV}/bin/activate
mkdir -p logs

# Keep BLAS inside the CPU allocation.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

echo "job ${SLURM_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none} on $(hostname)"
echo "data    : ${DAS_LFI_HOURLY_DIR}"
echo "results : ${DAS_LFI_OUTPUT_DIR}/bf_inv_disp"
