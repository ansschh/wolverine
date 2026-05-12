"""Route-level value model V(node) -- estimates cost-to-go to buyables.

RETRO_PLAN R-4. Retro*-style neural value, trained offline by
scripts/train_retro_value_model.py on AND-OR trees expanded with the
template proposer (supervision = realized cost / depth to a buyable leaf).

At inference, the planner consults V(node) to prioritise expansion of
the most promising OR_molecule node. When no checkpoint is loaded, the
fallback is a depth-only heuristic: V(node) = depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rasyn.synth.retro.schemas import RouteTreeNode


@dataclass
class ValueModelConfig:
    checkpoint_path: Path | None = None
    fallback_depth_weight: float = 1.0
    device: str = "cpu"


class ValueModel:
    def __init__(self, cfg: ValueModelConfig):
        self.cfg = cfg
        self._predict = None
        if cfg.checkpoint_path and cfg.checkpoint_path.exists():
            self._predict = self._load(cfg.checkpoint_path)

    def _load(self, ckpt_path: Path):
        try:
            import torch
        except ImportError:
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("predict_value")  # callable hook

    def value(self, node: RouteTreeNode) -> float:
        """Lower is better (cost-to-go semantics)."""
        if self._predict is not None and callable(self._predict):
            try:
                return float(self._predict(
                    smiles=node.molecule_smiles,
                    inchi_key=node.molecule_inchi_key,
                    depth=node.depth,
                ))
            except Exception:
                pass
        return float(node.depth) * self.cfg.fallback_depth_weight
