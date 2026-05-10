"""PyTorch concat-MLP ranker scaffold.

Architecture-agnostic per PLAN.md §17.A: starts as a concat-MLP at Layer-2
(small signal check), escalates to GNN/transformer at Layer-3 if the signal
warrants it. Multi-head: rescue_score (regression in [0,1]) + 7-class rescue
label + 6-class failure-mode multi-label + activity-retention bucket +
liability-improvement category.

Imports torch lazily so the rest of the package loads without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

EVIDENCE_FEATURE_DIMS = {
    "structural": 5,           # tanimoto, murcko_match, transformation_distance, mcs?, pharmacophore
    "descriptors": 9,          # MW, logP, TPSA, HBD, HBA, RotB, aromatic, fsp3, charge
    "descriptor_deltas": 9,
    "retention_bucket_oh": 5,  # strong/acceptable/weak/failed/unknown
    "improvement_oh": 6,       # large/moderate/minor/none/worse/unknown
    "rationale_count": 3,      # liability_drivers, modified, preserved counts
    "risk_flags": 3,           # new_liability_count, reactive_alert_count, has_synth_score
}
TOTAL_INPUT_DIM = sum(EVIDENCE_FEATURE_DIMS.values())


@dataclass
class ConcatMLPConfig:
    input_dim: int = TOTAL_INPUT_DIM
    hidden_dim: int = 512
    n_layers: int = 4
    dropout: float = 0.1
    n_rescue_labels: int = 7
    n_failure_modes: int = 6
    n_retention_buckets: int = 5
    n_improvement_categories: int = 6
    auxiliary_loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "rescue_score": 1.0,
            "rescue_label": 1.0,
            "failure_mode": 0.3,
            "activity_retention": 0.5,
            "liability_improvement": 0.5,
        }
    )


def build_concat_mlp_ranker(cfg: ConcatMLPConfig | None = None):
    """Build the concat-MLP ranker. Imports torch only when called."""
    import torch  # noqa: F401  (sanity-check import)
    import torch.nn as nn

    cfg = cfg or ConcatMLPConfig()

    class ConcatMLPRanker(nn.Module):
        def __init__(self):
            super().__init__()
            layers: list[nn.Module] = []
            d = cfg.input_dim
            for _ in range(cfg.n_layers):
                layers.extend([nn.Linear(d, cfg.hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout)])
                d = cfg.hidden_dim
            self.trunk = nn.Sequential(*layers)
            self.rescue_score_head = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
            self.rescue_label_head = nn.Linear(d, cfg.n_rescue_labels)
            self.failure_mode_head = nn.Linear(d, cfg.n_failure_modes)
            self.retention_head = nn.Linear(d, cfg.n_retention_buckets)
            self.improvement_head = nn.Linear(d, cfg.n_improvement_categories)
            self.confidence_head = nn.Linear(d, 5)  # overall, retention, liability, new_risk, plausibility

        def forward(self, x):
            h = self.trunk(x)
            return {
                "rescue_score": self.rescue_score_head(h).squeeze(-1),
                "rescue_label_logits": self.rescue_label_head(h),
                "failure_mode_logits": self.failure_mode_head(h),
                "retention_logits": self.retention_head(h),
                "improvement_logits": self.improvement_head(h),
                "confidence_logits": self.confidence_head(h),
            }

    return ConcatMLPRanker()
