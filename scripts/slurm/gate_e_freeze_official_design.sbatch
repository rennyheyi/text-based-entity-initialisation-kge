#!/bin/bash
#SBATCH --job-name=thesis-official-freeze
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=cluster_logs/gate-e-official-freeze-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the Thesis repository}"
mkdir -p cluster_logs reports configs

scratch_parent="${SLURM_TMPDIR:-/tmp}"
job_scratch="$(mktemp -d "${scratch_parent%/}/thesis-official-freeze-${USER}-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  case "$job_scratch" in
    "${scratch_parent%/}"/thesis-official-freeze-"${USER}"-"${SLURM_JOB_ID}".*)
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
python3 -m thesis.official_design \
  --protocol docs/PROTOCOL.md \
  --amendment docs/PROTOCOL_AMENDMENT_001.md \
  --claims-lock configs/official_claims_lock.json \
  --final-hyperparameters configs/final_hyperparameters.json \
  --hpo-design configs/hpo_design.json \
  --project-root "$PWD" \
  --scratch "$job_scratch" \
  --output-design configs/official_experiment_design.json \
  --output-report "reports/gate_e_official_design_freeze_${SLURM_JOB_ID}.json"

sha256sum \
  docs/PROTOCOL.md \
  docs/PROTOCOL_AMENDMENT_001.md \
  configs/official_claims_lock.json \
  configs/final_hyperparameters.json \
  configs/official_experiment_design.json \
  "reports/gate_e_official_design_freeze_${SLURM_JOB_ID}.json"
