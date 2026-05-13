#!/usr/bin/env bash
# Quick smoke that the env is wired up. Runs on login node (no GPU needed).
# ~30 seconds.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/resnick/scratch/atiwari2/rasyn-retro}"
VENV_DIR="${REPO_DIR}/.venv"

cd "${REPO_DIR}"

module purge
module load gcc/13.2.0-gcc-13.2.0-w55nxkl
module load python/3.11.6-gcc-11.3.1-rphh4kv
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${REPO_DIR}/rasyn"

echo "[smoke] full retro test suite (120 tests, ~1 second)"
python -m pytest tests/test_retro_*.py -q

echo "[smoke] curation smoke (200 USPTO-50K rows, no atom-mapping, no downloads-at-scale)"
python -m scripts.run_retro_curation \
  --smoke --skip-atom-mapping \
  --out-dir rasyn/data/clean/retro_smoke \
  --audit-dir artifacts/retro_decontam_audit_smoke

echo "[smoke] planner smoke (ibuprofen, no checkpoints, heuristic fallback)"
python -m scripts.run_retro_smoke --layer smoke --skip-forward-validation --time-budget-s 30

echo "[smoke] DONE"
