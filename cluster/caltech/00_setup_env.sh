#!/usr/bin/env bash
# Caltech HPC environment setup for Rasyn-Retro v1.
#
# Run ONCE on the login node (or inside an interactive compute job).
# Idempotent: skip steps that are already done.
#
# Modules confirmed available on Caltech HPC 2026-05-12:
#   cuda/12.2.1-gcc-11.3.1-sdqrj2e
#   python/3.11.6-gcc-11.3.1-rphh4kv
#   gcc/13.2.0-gcc-13.2.0-w55nxkl
#
# Account: tensorlab
# Scratch: /resnick/scratch/atiwari2/rasyn-retro/

set -euo pipefail

REPO_DIR="${REPO_DIR:-/resnick/scratch/atiwari2/rasyn-retro}"
VENV_DIR="${REPO_DIR}/.venv"

echo "[setup] REPO_DIR=${REPO_DIR}"
cd "${REPO_DIR}"

# --- Modules ---
echo "[setup] loading modules"
module purge
module load gcc/13.2.0-gcc-13.2.0-w55nxkl
module load python/3.11.6-gcc-11.3.1-rphh4kv
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e
python3 --version
which python3
gcc --version | head -1
nvcc --version | head -4 || echo "nvcc not on this node (login node OK)"

# --- Virtualenv (one per repo, lives on scratch) ---
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] creating venv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -V
pip install --upgrade pip wheel setuptools

# --- Core deps (pyproject) ---
echo "[setup] installing rasyn package (editable) + base deps"
cd "${REPO_DIR}/rasyn"
pip install -e ".[dev,chem,ml,data]" || pip install -e .
cd "${REPO_DIR}"

# --- Retro-specific deps (not in pyproject.toml) ---
echo "[setup] installing retro-specific deps"
pip install \
  "rxnmapper>=0.4" \
  "rdchiral>=1.1.0" \
  "ord-schema>=0.3" \
  "faiss-cpu>=1.8" \
  "numpy" "pandas" "pyarrow" "pydantic>=2.6" "pyyaml" "rdkit>=2024.3"

# --- PyTorch (CUDA 12.x build) ---
# Skip if already installed; pip will short-circuit otherwise.
echo "[setup] installing torch (CUDA 12 build)"
pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch>=2.3" "torchvision" "torchaudio" || \
  pip install "torch>=2.3"  # fallback to CPU build if cu121 index not reachable

# --- Sanity ---
echo "[setup] sanity import check"
python -c "
import sys, torch, rdkit, pydantic, pyarrow
from rasyn.synth.retro import schemas, registry
from rasyn.synth.retro.proposers import RetroProposer, TemplateProposer
print('python', sys.version.split()[0])
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(),
      'devices=', torch.cuda.device_count() if torch.cuda.is_available() else 0)
print('rdkit', rdkit.__version__)
print('pydantic', pydantic.__version__)
print('rasyn.synth.retro ok')
"

# --- Smoke pytest (CPU, no GPU needed) ---
echo "[setup] running retro pytest suite (120 tests, should all pass)"
cd "${REPO_DIR}/rasyn"
python -m pytest tests/test_retro_*.py -q
cd "${REPO_DIR}"

echo "[setup] DONE. To re-activate later:"
echo "  module load gcc/13.2.0-gcc-13.2.0-w55nxkl python/3.11.6-gcc-11.3.1-rphh4kv cuda/12.2.1-gcc-11.3.1-sdqrj2e"
echo "  source ${VENV_DIR}/bin/activate"
