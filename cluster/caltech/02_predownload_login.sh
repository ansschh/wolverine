#!/usr/bin/env bash
# Run on the Caltech HPC LOGIN NODE (which has clean outbound internet).
#
# Downloads every retro raw data source, validates it (no 0-byte files,
# no HTML-as-zip), and writes artifacts/retro_predownload_manifest.json
# with SHA256s. The R-1 sbatch refuses to start unless this manifest
# exists.
#
# Wall-time estimate (login node):
#   USPTO-50K  : ~30s (parquet, ~10 MB from HuggingFace)
#   USPTO-full : 30-90 min (Lowe 2017, ~3 GB tar.gz from figshare)
#   ZINC22     : 30-90 min (in-stock SMI, multi-GB gz)
#   Enamine BB : 5-15 min (~500 MB SDF)
#   eMolecules : 30-60 min (free tier, multi-GB gz)
# Total: ~1.5-4 hours depending on figshare/docking.org speed.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/resnick/scratch/atiwari2/rasyn-retro}"
VENV_DIR="${REPO_DIR}/.venv"
SOURCES="${SOURCES:-uspto_50k,uspto_full,zinc22,enamine,emolecules}"

cd "${REPO_DIR}"

module purge
module load gcc/13.2.0-gcc-13.2.0-w55nxkl
module load python/3.11.6-gcc-11.3.1-rphh4kv
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${REPO_DIR}/rasyn"

echo "[predownload] sources: ${SOURCES}"
echo "[predownload] raw root: rasyn/data/raw"
echo "[predownload] manifest: artifacts/retro_predownload_manifest.json"
echo

python -m scripts.predownload_retro_data \
  --raw-root rasyn/data/raw \
  --sources "${SOURCES}" \
  --manifest artifacts/retro_predownload_manifest.json

echo
echo "[predownload] DONE"
echo
echo "=== Manifest ==="
cat artifacts/retro_predownload_manifest.json
