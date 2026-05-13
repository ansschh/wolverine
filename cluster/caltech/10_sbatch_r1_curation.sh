#!/usr/bin/env bash
#SBATCH --job-name=retro-r1-curation
#SBATCH --account=tensorlab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=/resnick/scratch/atiwari2/rasyn-retro/logs/r1_curation_%j.out
#SBATCH --error=/resnick/scratch/atiwari2/rasyn-retro/logs/r1_curation_%j.err
#
# Rasyn-Retro Phase R-1: reaction data curation.
#   USPTO-50K (HF parquet) + USPTO-full (Lowe 2017, ~1.8M reactions)
#   Buyables: ZINC22 in-stock + Enamine REAL BB + eMolecules free
#   Canonicalize + RXNMapper atom-mapping + RDChiral template extraction
#   Decontaminate against sealed-case registry
#   Emit reactions_bronze/silver.parquet + buyables.parquet + templates.pkl
#
# USPTO-LLM and ORD are deliberately skipped here — they add 1.5-3 hours
# of downloads + multi-GB git clone for marginal R-2 gain. Add them back
# after R-6 ablation if conditions-head accuracy demands it.
#
# Wall-time estimate: ~3-5 hours on H100x4 once raw data is pre-staged.
# Pre-requisite: cluster/caltech/02_predownload_login.sh must have been
# run on the login node and produced artifacts/retro_predownload_manifest.json.
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

cd "${REPO_DIR}/rasyn"

echo "[r1] === preflight: verifying pre-downloaded raw data ==="
MANIFEST="artifacts/retro_predownload_manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "[r1] FATAL: ${MANIFEST} not found." >&2
  echo "[r1] Run cluster/caltech/02_predownload_login.sh on the LOGIN node first." >&2
  exit 2
fi
python - <<'PY'
import json, sys, zipfile, tarfile, gzip
from pathlib import Path

manifest_path = Path("artifacts/retro_predownload_manifest.json")
manifest = json.loads(manifest_path.read_text())
problems = []
for source, info in manifest.items():
    p = Path(info["path"])
    if not p.exists():
        problems.append(f"missing on disk: {p}")
        continue
    if p.stat().st_size != info["size_bytes"]:
        problems.append(
            f"size mismatch: {p} on-disk={p.stat().st_size} manifest={info['size_bytes']}"
        )
        continue
    # cheap content validation per extension
    name = p.name.lower()
    try:
        if name.endswith(".zip"):
            if not zipfile.is_zipfile(p):
                problems.append(f"not a zip: {p}")
        elif name.endswith(".tar.gz") or name.endswith(".tgz"):
            with tarfile.open(p, "r:gz") as tf:
                tf.next()
        elif name.endswith(".gz"):
            with gzip.open(p, "rb") as fh:
                fh.read(1)
        elif name.endswith(".parquet"):
            import pyarrow.parquet as pq
            pq.read_metadata(p)
    except Exception as e:
        problems.append(f"validation failed for {p}: {e}")
if problems:
    print("[r1] FATAL preflight problems:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    print("[r1] Re-run cluster/caltech/02_predownload_login.sh on the login node.", file=sys.stderr)
    sys.exit(2)
print(f"[r1] preflight: {len(manifest)} source(s) verified")
PY
echo "[r1] preflight OK"

echo "[r1] === smoke first (~30s, 200 USPTO-50K rows, no atom-mapping) ==="
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
  --rxnmapper-device cuda \
  --template-min-frequency 5 \
  --out-dir rasyn/data/clean/retro \
  --audit-dir artifacts/retro_decontam_audit

echo "[r1] === DONE date=$(date) ==="
ls -lh rasyn/data/clean/retro/
echo
cat artifacts/retro_decontam_audit/audit.json
