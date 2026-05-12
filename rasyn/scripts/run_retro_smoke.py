"""Three-layer smoke / slice / preflight verification for Rasyn-Retro (R-5).

Per RETRO_PLAN §6: every full run is preceded by smoke (1 target),
slice (~10 targets), preflight (~50 targets). Each layer asserts the
pipeline produces well-formed CandidateRoute objects and the planner
terminates within budget.

Run locally (no GPU, no trained checkpoints required — the planner
will use heuristic fallbacks):
    python scripts/run_retro_smoke.py --layer smoke

When checkpoints are available, point to them via flags:
    python scripts/run_retro_smoke.py --layer preflight \\
        --template-ckpt checkpoints/retro_template_v1/checkpoint.pt \\
        --templates rasyn/data/clean/retro/templates.pkl \\
        --buyables rasyn/data/clean/retro/buyables.parquet \\
        ...

Outputs:
    artifacts/retro_smoke/<layer>_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from rasyn.synth.retro.orchestrator import (
    CheckpointPaths, OrchestratorConfig, RetroOrchestrator,
)
from rasyn.synth.retro.planner import PlannerConfig

logger = logging.getLogger("retro.smoke")


SMOKE_TARGETS = [
    {"name": "ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "ikey": "HEFNNWSXXWATRW-UHFFFAOYSA-N"},
]

SLICE_TARGETS = SMOKE_TARGETS + [
    {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "ikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
    {"name": "acetaminophen", "smiles": "CC(=O)Nc1ccc(O)cc1", "ikey": "RZVAJINKPMORJF-UHFFFAOYSA-N"},
    {"name": "caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "ikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N"},
    {"name": "ranitidine", "smiles": "CNC(=Cc1cc(O)ccc1)NC(=S)NCCSCc1ccc(o1)CN(C)C",
     "ikey": "VMXUWOKSQNHOCA-UKTHLTGXSA-N"},
    {"name": "metformin", "smiles": "CN(C)C(=N)NC(=N)N", "ikey": "XZWYZXLIPXDOLR-UHFFFAOYSA-N"},
    {"name": "atorvastatin_frag", "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(c2ccccc2)c(c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
     "ikey": "XUKUURHRXDUEBC-UHFFFAOYSA-N"},
    {"name": "ethyl_acetate", "smiles": "CCOC(C)=O", "ikey": "XEKOWRVHYACXOJ-UHFFFAOYSA-N"},
    {"name": "biphenyl", "smiles": "c1ccccc1-c1ccccc1", "ikey": "ZUOUZKKEUPVFJK-UHFFFAOYSA-N"},
    {"name": "anisole", "smiles": "COc1ccccc1", "ikey": "RDOXTESZEPMUJZ-UHFFFAOYSA-N"},
]


def _build_orchestrator(args: argparse.Namespace) -> RetroOrchestrator:
    ckpt = CheckpointPaths(
        template=args.template_ckpt,
        templates_pickle=args.templates,
        graphedit=args.graphedit_ckpt,
        seq2seq=args.seq2seq_ckpt,
        retrieval_index=args.retrieval_index,
        retrieval_metadata=args.retrieval_metadata,
        diffusion=args.diffusion_ckpt,
        forward=args.forward_ckpt,
        conditions=args.conditions_ckpt,
        value=args.value_ckpt,
        buyables_parquet=args.buyables,
    )
    planner_cfg = PlannerConfig(
        max_steps=args.max_steps,
        max_iterations=args.max_iterations,
        time_budget_s=args.time_budget_s,
        top_k_per_proposer=args.top_k_per_proposer,
        top_k_routes=args.top_k_routes,
        enable_diffusion=not args.disable_diffusion,
        require_forward_validation=not args.skip_forward_validation,
        tier1_only=args.tier1_only,
    )
    return RetroOrchestrator(OrchestratorConfig(
        planner=planner_cfg, ckpt=ckpt,
        enable_proposers=args.enable_proposers,
    ))


def _run_layer(orch: RetroOrchestrator, targets: list[dict], time_budget_s: float) -> dict:
    summary = {"n_targets": len(targets), "results": []}
    t0 = time.time()
    for tgt in targets:
        t1 = time.time()
        candidates = orch.plan_routes(target_smiles=tgt["smiles"], target_inchi_key=tgt["ikey"])
        elapsed = time.time() - t1
        summary["results"].append({
            "name": tgt["name"],
            "smiles": tgt["smiles"],
            "n_candidates": len(candidates),
            "top_route_score": candidates[0].route_score if candidates else None,
            "top_step_count": (
                candidates[0].route_tree.step_count if candidates else None
            ),
            "all_leaves_buyable": (
                candidates[0].route_tree.all_leaves_buyable if candidates else False
            ),
            "elapsed_s": elapsed,
        })
        if time.time() - t0 > time_budget_s:
            logger.warning("layer time budget exhausted; stopping early")
            break
    summary["total_elapsed_s"] = time.time() - t0
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", choices=["smoke", "slice", "preflight"], default="smoke")
    p.add_argument("--out", type=Path, default=Path("artifacts/retro_smoke"))
    # Checkpoint paths (all optional; planner uses heuristic fallback when missing)
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
    # Planner config
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-iterations", type=int, default=200)
    p.add_argument("--time-budget-s", type=float, default=60.0)
    p.add_argument("--top-k-per-proposer", type=int, default=5)
    p.add_argument("--top-k-routes", type=int, default=10)
    p.add_argument("--disable-diffusion", action="store_true")
    p.add_argument("--skip-forward-validation", action="store_true")
    p.add_argument("--tier1-only", action="store_true")
    p.add_argument("--enable-proposers", nargs="+",
                   default=["template", "graphedit", "seq2seq", "retrieval", "diffusion"])
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    targets = {"smoke": SMOKE_TARGETS, "slice": SLICE_TARGETS, "preflight": SLICE_TARGETS}[args.layer]
    time_budget = {"smoke": 60.0, "slice": 600.0, "preflight": 3600.0}[args.layer]

    logger.info("layer=%s targets=%d", args.layer, len(targets))
    orch = _build_orchestrator(args)
    summary = _run_layer(orch, targets, time_budget)

    out_path = args.out / f"{args.layer}_report.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", out_path)

    # Smoke / slice / preflight pass criteria
    if args.layer == "smoke":
        ok = summary["total_elapsed_s"] < 60.0
        logger.info("smoke %s in %.1fs", "OK" if ok else "FAIL", summary["total_elapsed_s"])
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
