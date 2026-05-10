"""MoleculeNet adapter via DeepChem.

MoleculeNet is used purely for auxiliary property pretraining and as
sanity-check benchmark. NOT a primary rescue-pair source.
"""

from __future__ import annotations

DATASETS = (
    "tox21",
    "toxcast",
    "muv",
    "hiv",
    "bace_classification",
    "bbbp",
    "delaney",  # ESOL solubility
    "lipo",
    "freesolv",
    "qm9",
    "sider",
)


def load_dataset(name: str, *, data_dir: str = "rasyn/data/raw/molnet"):
    """Load a MoleculeNet dataset via DeepChem.

    Requires `pip install -e '.[data]'` (deepchem). Returns a DeepChem dataset
    triple (train, valid, test) — see DeepChem docs.
    """
    import deepchem as dc  # type: ignore[import-not-found]

    loader_name = f"load_{name}"
    if not hasattr(dc.molnet, loader_name):
        raise ValueError(f"Unknown MoleculeNet dataset: {name}")
    loader = getattr(dc.molnet, loader_name)
    tasks, datasets, transformers = loader(data_dir=data_dir, save_dir=data_dir)
    return {"tasks": tasks, "datasets": datasets, "transformers": transformers}
