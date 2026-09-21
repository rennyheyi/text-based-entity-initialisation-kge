#!/bin/bash
#SBATCH --job-name=thesis-hpo-s1
#SBATCH --partition=compute
#SBATCH --exclude=gpu-nvidia-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-71%2
#SBATCH --output=cluster_logs/hpo-stage1-%A-%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the Thesis repository}"
mkdir -p cluster_logs reports runs/hpo/stage1

scratch_parent="${SLURM_TMPDIR:-/tmp}"
job_scratch="$(mktemp -d "${scratch_parent%/}/thesis-hpo-stage1-${USER}-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  case "$job_scratch" in
    "${scratch_parent%/}"/thesis-hpo-stage1-"${USER}"-"${SLURM_JOB_ID}".*)
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Allocation confirmed: job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID} host=$(hostname)"
python3 -m unittest discover -s tests -v
python3 -m thesis.hpo_stage1 \
  --design configs/hpo_design.json \
  --preflight-report reports/gate_c_hardware_preflight_440725.json \
  --protocol docs/PROTOCOL.md \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --project-root "$PWD" \
  --scratch "$job_scratch" \
  --artifact-root runs/hpo/stage1 \
  --output "reports/hpo_stage1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
