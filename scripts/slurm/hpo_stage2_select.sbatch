#!/bin/bash
#SBATCH --job-name=thesis-hpo-s2-select
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=cluster_logs/hpo-stage2-select-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the Thesis repository}"
mkdir -p cluster_logs reports configs

scratch_parent="${SLURM_TMPDIR:-/tmp}"
job_scratch="$(mktemp -d "${scratch_parent%/}/thesis-hpo-stage2-select-${USER}-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  case "$job_scratch" in
    "${scratch_parent%/}"/thesis-hpo-stage2-select-"${USER}"-"${SLURM_JOB_ID}".*)
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
python3 -m thesis.hpo_stage2_selection \
  --stage2-design configs/hpo_stage2_design.json \
  --source-design configs/hpo_design.json \
  --protocol docs/PROTOCOL.md \
  --attempt-lock configs/hpo_stage2_attempt_lock.json \
  --reports-dir reports \
  --project-root "$PWD" \
  --artifact-root runs/hpo/stage2 \
  --scratch "$job_scratch" \
  --stage3-design-output configs/hpo_stage3_design.json \
  --output "reports/hpo_stage2_selection_${SLURM_JOB_ID}.json"
