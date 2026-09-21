#!/usr/bin/env bash
#SBATCH --job-name=thesis-gate-a
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=cluster_logs/gate-a-source-audit-%j.out

set -euo pipefail
umask 077

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
mkdir -p cluster_logs reports

scratch_root="${SLURM_TMPDIR:-/tmp}"
THESIS_JOB_SCRATCH="$(mktemp -d -p "$scratch_root" "thesis-gate-a-${USER}-${SLURM_JOB_ID}-XXXXXX")"
export THESIS_JOB_SCRATCH

cleanup() {
  case "$THESIS_JOB_SCRATCH" in
    "${scratch_root%/}"/thesis-gate-a-"${USER}"-"${SLURM_JOB_ID}"-*)
      rm -rf -- "$THESIS_JOB_SCRATCH"
      ;;
    *)
      echo "Refusing to remove unexpected scratch path: $THESIS_JOB_SCRATCH" >&2
      ;;
  esac
}
trap cleanup EXIT

export PYTHONPATH="$PWD/src"
export PYTHONHASHSEED=0

echo "Allocation confirmed: job=${SLURM_JOB_ID} host=$(hostname)"
python3 -m unittest discover -s tests -v
python3 -m thesis.source_audit \
  --config configs/source_candidates.json \
  --scratch "$THESIS_JOB_SCRATCH" \
  --output "reports/gate_a_source_audit_${SLURM_JOB_ID}.json"
