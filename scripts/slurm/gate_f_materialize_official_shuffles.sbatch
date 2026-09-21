#!/bin/bash
#SBATCH --job-name=thesis-official-shuffles
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=cluster_logs/gate-f-official-shuffles-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the Thesis repository}"
mkdir -p cluster_logs reports data/manifests

scratch_parent="${SLURM_TMPDIR:-/tmp}"
job_scratch="$(mktemp -d "${scratch_parent%/}/thesis-official-shuffles-${USER}-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  case "$job_scratch" in
    "${scratch_parent%/}"/thesis-official-shuffles-"${USER}"-"${SLURM_JOB_ID}".*)
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

echo "Allocation confirmed: job=${SLURM_JOB_ID} host=$(hostname)"
python3 -m unittest discover -s tests -v
python3 -m thesis.official_shuffle \
  --design configs/official_experiment_design.json \
  --project-root "$PWD" \
  --scratch "$job_scratch" \
  --output "reports/gate_f_official_shuffles_${SLURM_JOB_ID}.json"
