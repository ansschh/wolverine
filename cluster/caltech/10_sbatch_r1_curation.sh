#!/usr/bin/env bash
#SBATCH --job-name=retro-r1-curation
#SBATCH --account=tensorlab
#SBATCH --partition=sunshine
#SBATCH --nodelist=hpc-79-11
#SBATCH --gres=gpu:v100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00
#SBATCH --output=/resnick/scratch/atiwari2/rasyn-retro/logs/r1_curation_%j.out
#SBATCH --error=/resnick/scratch/atiwari2/rasyn-retro/logs/r1_curation_%j.err
#
# Rasyn-Retro Phase R-1: reaction data curation.
#   USPTO 50K + full + LLM (~1.8M reactions)
#   ORD (open-reaction-database/ord-data git clone)
#   Buyables: ZINC22 in-stock + Enamine REAL BB + eMolecules free
#   Canonicalize + RXNMapper atom-mapping + RDChiral template extraction
#   Decontaminate against sealed-case registry
#   Emit reactions_bronze/silver.parquet + buyables.parquet + templates.pkl
#
# Wall-time estimate: 8-16 hours on V100x8.
# Output: rasyn/data/clean/retro/ + artifacts/retro_decontam_audit/

set -euo pipefail

REPO_DIR="/resnick/scratch/atiwari2/rasyn-retro"
VENV_DIR="${REPO_DIR}/.venv"

cd "${REPO_DIR}"
mkdir -p logs

echo "[r1] === environment ==="
echo "[r1] node=$(hostname) date=$(date)"
echo "[r1] SLURM_JOB_ID=${SLURM_JOB_ID}"
nvidia-smi || true

module purge
module load gcc/13.2.0-gcc-13.2.0-w55nxkl
module load python/3.11.6-gcc-11.3.1-rphh4kv
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[r1] === smoke first (~30s, 200 USPTO-50K rows, no atom-mapping) ==="
cd "${REPO_DIR}/rasyn"
python -m scripts.run_retro_curation \
  --smoke \
  --skip-atom-mapping \
  --out-dir rasyn/data/clean/retro_smoke \
  --audit-dir artifacts/retro_decontam_audit_smoke
echo "[r1] smoke OK"

echo "[r1] === full R-1 curation ==="
python -m scripts.run_retro_curation \
  --use-uspto-50k \
  --use-uspto-full \
  --use-uspto-llm \
  --use-ord \
  --rxnmapper-device cuda \
  --template-min-frequency 5 \
  --out-dir rasyn/data/clean/retro \
  --audit-dir artifacts/retro_decontam_audit

echo "[r1] === DONE date=$(date) ==="
ls -lh rasyn/data/clean/retro/
echo
cat artifacts/retro_decontam_audit/audit.json
