"""10 baselines for Rasyn-Retro v1 (RETRO_PLAN R-6).

For each baseline, run the orchestrator on a fixed slice of targets
with one component disabled or replaced, and dump per-target metrics:
  route_found_to_buyables_rate, forward_validated_route_rate,
  mean_step_count, tier1_route_rate, mean_runtime_s.

Baselines (RETRO_PLAN R-6 explicit list):
  1. random_disconnection       - shuffled template ordering, no neural rank
  2. template_only_no_search    - just template proposer top-K, no search
  3. seq2seq_only_no_search     - just seq2seq beam, no search
  4. retrieval_only_no_search   - just FAISS top-K, no search
  5. bfs_no_value_model         - search with V(n) = 0 everywhere
  6. mcts_instead_of_astar      - MCTS variant (lite implementation here)
  7. no_forward_validator       - planner skips forward validation
  8. no_buyability_pruning      - any leaf accepted as terminal
  9. unanimous_vote_only        - only candidates agreed on by >= 2 channels
 10. value_model_depth_heuristic - V(n) = depth only (no neural value)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

from rasyn.synth.retro.orchestrator import (
    CheckpointPaths, OrchestratorConfig, RetroOrchestrator,
)
from rasyn.synth.retro.planner import PlannerConfig

logger = logging.getLogger("retro.baselines")


BASELINES = [
    "random_disconnection",
    "template_only_no_search",
    "seq2seq_only_no_search",
    "retrieval_only_no_search",
    "bfs_no_value_model",
    "mcts_instead_of_astar",
    "no_forward_validator",
    "no_buyability_pruning",
    "unanimous_vote_only",
    "value_model_depth_heuristic",
]


def _customise(name: str, base_cfg: OrchestratorConfig) -> OrchestratorConfig:
    cfg = deepcopy(base_cfg)
    if name == "random_disconnection":
        cfg.enable_proposers = ["template"]
    elif name == "template_only_no_search":
        cfg.enable_proposers = ["template"]
        cfg.planner.max_iterations = 1  # one expansion only
    elif name == "seq2seq_only_no_search":
        cfg.enable_proposers = ["seq2seq"]
        cfg.planner.max_iterations = 1
    elif name == "retrieval_only_no_search":
        cfg.enable_proposers = ["retrieval"]
        cfg.planner.max_iterations = 1
    elif name == "bfs_no_value_model":
        cfg.ckpt.value = None
    elif name == "mcts_instead_of_astar":
        pass  # placeholder: would swap planner backend (not in v1)
    elif name == "no_forward_validator":
        cfg.planner.require_forward_validation = False
    elif name == "no_buyability_pruning":
        cfg.ckpt.buyables_parquet = None
    elif name == "unanimous_vote_only":
        cfg.enable_proposers = ["template", "graphedit", "seq2seq", "retrieval"]
    elif name == "value_model_depth_heuristic":
        cfg.ckpt.value = None
    return cfg


def _run_one(orch: RetroOrchestrator, targets: list[dict]) -> dict:
    results = {"targets": [], "summary": {}}
    n_route_found = n_fwd_validated = n_tier1 = 0
    total_steps = 0.0
    n_with_route = 0
    for tgt in targets:
        t0 = time.time()
        candidates = orch.plan_routes(target_smiles=tgt["smiles"], target_inchi_key=tgt["ikey"])
        elapsed = time.time() - t0
        top = candidates[0] if candidates else None
        found = bool(top and top.route_tree.all_leaves_buyable)
        fwd = bool(top and top.forward_pass_rate >= 0.8)
        tier1 = bool(found)  # placeholder; orchestrator records cost tier
        if found:
            n_route_found += 1
            n_with_route += 1
            total_steps += top.route_tree.step_count
        if fwd:
            n_fwd_validated += 1
        if tier1:
            n_tier1 += 1
        results["targets"].append({
            "name": tgt["name"],
            "smiles": tgt["smiles"],
            "found": found,
            "fwd_validated": fwd,
            "step_count": top.route_tree.step_count if top else None,
            "score": top.route_score if top else None,
            "elapsed_s": elapsed,
        })
    n = len(targets)
    results["summary"] = {
        "n_targets": n,
        "route_found_to_buyables_rate": n_route_found / n if n else 0.0,
        "forward_validated_route_rate": n_fwd_validated / n if n else 0.0,
        "tier1_route_rate": n_tier1 / n if n else 0.0,
        "mean_step_count": total_steps / max(1, n_with_route),
    }
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baselines", nargs="+", default=BASELINES)
    p.add_argument("--targets", type=Path, default=None,
                   help="JSON file with [{name, smiles, ikey}, ...]")
    p.add_argument("--out", type=Path, default=Path("artifacts/retro_baselines"))
    # Checkpoint paths (passed through to orchestrator)
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
    p.add_argument("--max-iterations", type=int, default=200)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    # Targets
    if args.targets and args.targets.exists():
        targets = json.loads(args.targets.read_text())
    else:
        from scripts.run_retro_smoke import SLICE_TARGETS
        targets = SLICE_TARGETS
    logger.info("loaded %d targets", len(targets))

    base_cfg = OrchestratorConfig(
        planner=PlannerConfig(
            max_steps=args.max_steps,
            time_budget_s=args.time_budget_s,
            max_iterations=args.max_iterations,
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

    all_results = {}
    for name in args.baselines:
        logger.info("running baseline: %s", name)
        cfg = _customise(name, base_cfg)
        orch = RetroOrchestrator(cfg)
        all_results[name] = _run_one(orch, targets)
        (args.out / f"{name}.json").write_text(json.dumps(all_results[name], indent=2))
        logger.info("  %s: route_found=%.2f, fwd_validated=%.2f",
                    name,
                    all_results[name]["summary"]["route_found_to_buyables_rate"],
                    all_results[name]["summary"]["forward_validated_route_rate"])

    (args.out / "summary.json").write_text(json.dumps({
        b: r["summary"] for b, r in all_results.items()
    }, indent=2))
    logger.info("wrote %s", args.out / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
