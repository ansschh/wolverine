"""Ablations for Rasyn-Retro v1 (RETRO_PLAN R-6).

Each ablation removes ONE component of the production system and
measures how much performance drops on the slice set. Useful to argue
that each component carries its weight.

Ablations:
  - no_template
  - no_graphedit
  - no_seq2seq
  - no_retrieval
  - no_diffusion              (compare to L3 diffusion-in-v1 honesty floor)
  - no_conditions
  - no_value_model
  - no_film_conditioning      (template / seq2seq trained without FiLM)
  - no_hard_negatives         (template trained without hard negatives)
  - single_seed_only          (no ensemble)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

from rasyn.synth.retro.orchestrator import (
    CheckpointPaths, OrchestratorConfig, RetroOrchestrator,
)
from rasyn.synth.retro.planner import PlannerConfig

logger = logging.getLogger("retro.ablations")


ABLATIONS = [
    "no_template",
    "no_graphedit",
    "no_seq2seq",
    "no_retrieval",
    "no_diffusion",
    "no_conditions",
    "no_value_model",
    "no_film_conditioning",
    "no_hard_negatives",
    "single_seed_only",
]


def _customise_ablation(name: str, base_cfg: OrchestratorConfig) -> OrchestratorConfig:
    cfg = deepcopy(base_cfg)
    if name == "no_template":
        cfg.enable_proposers = [p for p in cfg.enable_proposers if p != "template"]
    elif name == "no_graphedit":
        cfg.enable_proposers = [p for p in cfg.enable_proposers if p != "graphedit"]
    elif name == "no_seq2seq":
        cfg.enable_proposers = [p for p in cfg.enable_proposers if p != "seq2seq"]
    elif name == "no_retrieval":
        cfg.enable_proposers = [p for p in cfg.enable_proposers if p != "retrieval"]
    elif name == "no_diffusion":
        cfg.enable_proposers = [p for p in cfg.enable_proposers if p != "diffusion"]
    elif name == "no_conditions":
        cfg.ckpt.conditions = None
    elif name == "no_value_model":
        cfg.ckpt.value = None
    elif name == "no_film_conditioning":
        # Would point at FiLM-stripped checkpoints; placeholder for v1.
        pass
    elif name == "no_hard_negatives":
        # Would point at template ckpt trained without hard-negative mining.
        pass
    elif name == "single_seed_only":
        # Use only the seed-42 ckpt; placeholder.
        pass
    return cfg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ablations", nargs="+", default=ABLATIONS)
    p.add_argument("--targets", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("artifacts/retro_ablations"))
    p.add_argument("--template-ckpt", type=Path, default=None)
    p.add_argument("--templates", type=Path, default=None)
    p.add_argument("--graphedit-ckpt", type=Path, default=None)
    p.add_argument("--seq2seq-ckpt", type=Path, default=None)
    p.add_argument("--retrieval-index", type=Path, default=None)
    p.add_argument("--retrieval-metadata", type=Path, default=None)
    p.add_argument("--diffusion-ckpt", type=Path, default=None)
    p.add_argument("--forward-ckpt", type=Path, default=None)
    p.add_argument("--conditions-ckpt", type=Path, default=None)
    p.add_argument("--value-ckpt", type=Path, default=None)
    p.add_argument("--buyables", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--time-budget-s", type=float, default=120.0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.targets and args.targets.exists():
        targets = json.loads(args.targets.read_text())
    else:
        from scripts.run_retro_smoke import SLICE_TARGETS
        targets = SLICE_TARGETS

    base_cfg = OrchestratorConfig(
        planner=PlannerConfig(
            max_steps=args.max_steps,
            time_budget_s=args.time_budget_s,
        ),
        ckpt=CheckpointPaths(
            template=args.template_ckpt, templates_pickle=args.templates,
            graphedit=args.graphedit_ckpt, seq2seq=args.seq2seq_ckpt,
            retrieval_index=args.retrieval_index, retrieval_metadata=args.retrieval_metadata,
            diffusion=args.diffusion_ckpt, forward=args.forward_ckpt,
            conditions=args.conditions_ckpt, value=args.value_ckpt,
            buyables_parquet=args.buyables,
        ),
    )
    from scripts.run_retro_baselines import _run_one  # reuse harness

    summary = {}
    for name in args.ablations:
        logger.info("running ablation: %s", name)
        cfg = _customise_ablation(name, base_cfg)
        orch = RetroOrchestrator(cfg)
        result = _run_one(orch, targets)
        summary[name] = result["summary"]
        (args.out / f"{name}.json").write_text(json.dumps(result, indent=2))
        logger.info("  %s: route_found=%.2f", name,
                    result["summary"]["route_found_to_buyables_rate"])

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
