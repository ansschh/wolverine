# Caltech HPC deployment scripts for Rasyn-Retro

Target: Resnick High Performance Computing Center.
Account: `tensorlab`. Scratch: `/resnick/scratch/atiwari2/rasyn-retro/`.

## Quickstart

```bash
# Login node, one-time
cd /resnick/scratch/atiwari2/rasyn-retro
bash cluster/caltech/00_setup_env.sh         # ~5-10 min: modules + venv + pip + pytest

# Login node — pre-download raw data (compute nodes have restricted egress)
bash cluster/caltech/02_predownload_login.sh # ~1.5-4 h depending on figshare/zinc speed

# Validate (login node)
bash cluster/caltech/01_smoke_local.sh       # ~30s: imports + curation smoke + planner smoke

# Launch R-1 (data curation) on the gpu partition (H100x4)
sbatch cluster/caltech/10_sbatch_r1_curation.sh
squeue -u $USER
```

Logs land at `/resnick/scratch/atiwari2/rasyn-retro/logs/r1_curation_<jobid>.{out,err}`.

The sbatch preflight refuses to start if `artifacts/retro_predownload_manifest.json`
is missing or any listed archive fails validation (size, zip/tar/gz integrity).

## Module stack (verified 2026-05-12)

| Module | Tag |
|---|---|
| `gcc/13.2.0` | `gcc-13.2.0-w55nxkl` |
| `python/3.11.6` | `gcc-11.3.1-rphh4kv` |
| `cuda/12.2.1` | `gcc-11.3.1-sdqrj2e` |

## Pip deps (installed by `00_setup_env.sh`)

- `torch>=2.3` (CUDA 12.1 build via PyTorch's `download.pytorch.org/whl/cu121`)
- `rdkit>=2024.3`
- `rxnmapper>=0.4`
- `rdchiral>=1.1.0`
- `ord-schema>=0.3`
- `faiss-cpu>=1.8`
- `pydantic>=2.6`, `pyarrow`, `numpy`, `pandas`, `pyyaml`

## Phase mapping

| Phase | Script | Pod | Time |
|---|---|---|---|
| Pre-download raw data (login node) | `02_predownload_login.sh` | login node, no GPU | 1.5–4 h |
| R-1 data curation | `10_sbatch_r1_curation.sh` | H100 x4, gpu partition | 3–5 h |
| R-2 template proposer | TBD | A100/H100 | 4–8 h |
| R-2 graphedit | TBD | A100/H100 | 12–24 h |
| R-2 seq2seq | TBD | A100/H100 x4-5 | 24–48 h |
| R-2 diffusion | TBD | A100/H100 x5 | 48–96 h |
| R-2 retrieval index | (CPU, runs alongside R-1) | CPU | 10–30 min |
| R-3 forward + conditions | TBD | A100/H100 x5 | 24–48 h |
| R-4 value model | TBD | A100/H100 x2 | 16–32 h |
| R-5 planner smoke / slice / preflight | TBD | 1 GPU | 4–8 h |
| R-6 baselines + ablations | TBD | A100/H100 x2 | 8–16 h |
| R-7 sealed cases | TBD | A100/H100 x2 | 4–8 h |

R-2…R-7 sbatch scripts will be added after R-1 lands real data.

## Resume after disconnect

The Caltech `gpu` partition has a 14-day time-limit; `sunshine`/`dgxlo` are infinite. Jobs survive
SSH disconnects. To check on a job:

```bash
squeue -j <jobid>
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,NodeList
tail -f /resnick/scratch/atiwari2/rasyn-retro/logs/r1_curation_<jobid>.out
```

## Storage

- Code: `/resnick/scratch/atiwari2/rasyn-retro/` (git tracked)
- Data: `/resnick/scratch/atiwari2/rasyn-retro/rasyn/data/` (NOT git tracked, recreatable)
- Checkpoints: `/resnick/scratch/atiwari2/rasyn-retro/checkpoints/` (NOT git tracked)
- Logs: `/resnick/scratch/atiwari2/rasyn-retro/logs/`

VAST scratch has 401 TB free. Retro needs ~50–100 GB.
