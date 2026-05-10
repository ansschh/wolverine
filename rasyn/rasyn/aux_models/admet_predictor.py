"""Multi-task ADMET predictor scaffold.

One backbone shared across the 22 TDC ADMET datasets + the 4 rescue-relevant
liability families. Heads:
  - hERG   IC50 (regression) + risk-category (classification)
  - solubility logS (regression)
  - metabolic_stability half-life / clearance (regression)
  - oral_exposure bioavailability (regression)
  - permeability Caco-2 (regression)
  - cytotoxicity Tox21 (multi-task BCE)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ADMETPredictorConfig:
    backbone: str = "transformer"
    hidden_dim: int = 512
    n_layers: int = 6
    dropout: float = 0.1


def build_admet_predictor(cfg: ADMETPredictorConfig | None = None):
    """Build the multi-task ADMET predictor (PyTorch nn.Module). Imports torch lazily."""
    import torch.nn as nn

    cfg = cfg or ADMETPredictorConfig()

    class ADMETPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.hidden_dim
            self.trunk = nn.Sequential(
                *[nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout)) for _ in range(cfg.n_layers)]
            )
            self.heads = nn.ModuleDict(
                {
                    "hERG_IC50": nn.Linear(d, 1),
                    "hERG_risk": nn.Linear(d, 4),
                    "solubility_logS": nn.Linear(d, 1),
                    "halflife": nn.Linear(d, 1),
                    "clearance": nn.Linear(d, 1),
                    "bioavailability": nn.Linear(d, 1),
                    "permeability": nn.Linear(d, 1),
                    "tox21": nn.Linear(d, 12),
                }
            )

        def forward(self, h):
            h = self.trunk(h)
            return {name: head(h) for name, head in self.heads.items()}

    return ADMETPredictor()
