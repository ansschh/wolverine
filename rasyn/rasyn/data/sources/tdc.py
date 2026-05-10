"""TDC (Therapeutics Data Commons) adapter for auxiliary ADMET endpoints.

TDC ships 22 ADMET datasets via `tdc.single_pred.ADME` and `tdc.single_pred.Tox`.
We use them to (a) train auxiliary "clean" predictors that feed the evidence
builder, and (b) cross-check ChEMBL liability values.

CRITICAL: TDC datasets may include the sealed-case answer molecules
(fexofenadine, valacyclovir likely present in solubility/permeability sets).
EVERY TDC load must run through `rasyn.data.decontam.quarantine.scrub` against
the sealed-case registry before being used as training data.
"""

from __future__ import annotations

from dataclasses import dataclass

ADME_DATASETS = (
    "Caco2_Wang",
    "PAMPA_NCATS",
    "HIA_Hou",
    "Pgp_Broccatelli",
    "Bioavailability_Ma",
    "Lipophilicity_AstraZeneca",
    "Solubility_AqSolDB",
    "HydrationFreeEnergy_FreeSolv",
    "BBB_Martins",
    "PPBR_AZ",
    "VDss_Lombardo",
    "CYP2C19_Veith",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
    "CYP1A2_Veith",
    "CYP2C9_Veith",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
)

TOX_DATASETS = (
    "hERG",
    "hERG_Karim",
    "AMES",
    "DILI",
    "Skin_Reaction",
    "Carcinogens_Lagunin",
    "ClinTox",
    "LD50_Zhu",
    "Tox21",
    "ToxCast",
)


@dataclass
class TDCConfig:
    cache_dir: str = "rasyn/data/raw/tdc"


def load_dataset(name: str, *, cfg: TDCConfig | None = None):
    """Load one TDC dataset as a pandas DataFrame.

    Requires `pip install -e '.[data]'`. Imports lazily so the rest of the
    package loads without TDC.
    """
    from tdc.single_pred import ADME, Tox  # type: ignore[import-not-found]

    cfg = cfg or TDCConfig()
    if name in ADME_DATASETS:
        ds = ADME(name=name, path=cfg.cache_dir)
    elif name in TOX_DATASETS:
        ds = Tox(name=name, path=cfg.cache_dir)
    else:
        raise ValueError(f"Unknown TDC dataset: {name}")
    return ds.get_data()


def liability_relevant_datasets(liability_type: str) -> list[str]:
    """Map a Rasyn liability_type to the relevant TDC datasets."""
    mapping = {
        "hERG": ["hERG", "hERG_Karim"],
        "solubility": ["Solubility_AqSolDB", "HydrationFreeEnergy_FreeSolv", "Lipophilicity_AstraZeneca"],
        "metabolic_stability": ["Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"],
        "oral_exposure": ["Bioavailability_Ma", "HIA_Hou", "Caco2_Wang", "PAMPA_NCATS"],
        "cyp_inhibition": ["CYP2C19_Veith", "CYP2D6_Veith", "CYP3A4_Veith", "CYP1A2_Veith", "CYP2C9_Veith"],
        "permeability": ["Caco2_Wang", "PAMPA_NCATS"],
        "cytotoxicity": ["Tox21", "ToxCast"],
    }
    return mapping.get(liability_type, [])
