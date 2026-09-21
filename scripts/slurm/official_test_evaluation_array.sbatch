#!/bin/bash
#SBATCH --job-name=thesis-official-test
#SBATCH --partition=compute
#SBATCH --exclude=gpu-nvidia-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=cluster_logs/official-test-%A-%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the Thesis repository}"
mkdir -p cluster_logs reports/official_test artifacts/official_test

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This runner requires a Slurm array task" >&2
  exit 1
fi

scratch_parent="${SLURM_TMPDIR:-/tmp}"
job_scratch="$(mktemp -d "${scratch_parent%/}/thesis-official-test-${USER}-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  case "$job_scratch" in
    "${scratch_parent%/}"/thesis-official-test-"${USER}"-"${SLURM_JOB_ID}".*)
      rm -rf -- "$job_scratch"
      ;;
    *)
      echo "Refusing to clean unexpected scratch path: $job_scratch" >&2
      return 1
      ;;
  esac
}
trap cleanup EXIT

conda_script="/storage/home/${USER}/miniforge3/etc/profile.d/conda.sh"
if [[ ! -f "$conda_script" ]]; then
  echo "Conda activation script is missing: $conda_script" >&2
  exit 1
fi
source "$conda_script"
conda activate kg-thesis

export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

echo "Allocation confirmed: job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
python3 -m thesis.official_test_evaluation \
  --design configs/official_experiment_design.json \
  --checkpoint-lock configs/official_checkpoint_lock.json \
  --evaluation-lock configs/official_evaluation_lock.json \
  --calibration-lock configs/official_calibration_lock.json \
  --claims-lock configs/official_claims_lock.json \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --project-root "$PWD" \
  --scratch "$job_scratch"
