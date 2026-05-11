"""ABX Channel F: organism-conditioned, narrow-spectrum-selective generator.

Same encoder-decoder architecture as Channel E, but trained on the SUBSET of
generative examples whose `full_molecule_inchi_key` is active against ≤ 2
organisms in the gen-examples table (pathogen-specific / narrow-spectrum).

This is a tighter teacher distribution than Channel E (which trains on all
active molecules across all organisms, including broad-spectrum compounds).
Same role-relationship as ADMET Ch5 vs Ch4: a more conservative, more
selective generator from the same architecture.

Per L41: not a placeholder for graph diffusion — a real trained seq2seq
producing organism-selective candidates.

Run on Pod D (8x A100 DDP):
    cd ~/wolverine/rasyn
    torchrun --nproc_per_node=8 --standalone scripts/train_abx_channel_f.py \\
        --pretrain rasyn/data/clean/smiles_lm_200m/checkpoint.pt \\
        --gen-examples rasyn/data/clean/antibiotic/generative_training_examples.parquet \\
        --steps 4000 --bs 32 --lr 1e-4 --seed 43 \\
        --out rasyn/data/clean/abx_channel_f
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import train_abx_channel_e as ch_e  # noqa: E402


def filter_examples_f(df: pd.DataFrame) -> pd.DataFrame:
    """Channel F filter: active rows for narrow-spectrum molecules only.

    Definition (operational): a molecule is narrow-spectrum if its
    `full_molecule_inchi_key` appears as 'active' against ≤ 2 distinct
    organisms in the generative examples table.
    """
    df = df[df["full_molecule_smiles"].notna()]
    df = df[df["organism_context"].notna()]
    df = df[df["activity_label"] == "active"]
    if "full_molecule_inchi_key" in df.columns:
        org_counts = (
            df.groupby("full_molecule_inchi_key")["organism_context"].nunique()
        )
        narrow_keys = set(org_counts[org_counts <= 2].index)
        df = df[df["full_molecule_inchi_key"].isin(narrow_keys)]
    return df


def _patched_save(model, args, step, channel_marker: str):
    """Override channel marker for Ch-F."""
    return _orig_save(model, args, step, "abx_channel_f_narrow_spectrum")


# Patch in Ch-F filter + channel marker
_orig_save = ch_e._save
ch_e._save = _patched_save


if __name__ == "__main__":
    sys.exit(ch_e.main(filter_fn=filter_examples_f, channel_marker="abx_channel_f_narrow_spectrum"))
