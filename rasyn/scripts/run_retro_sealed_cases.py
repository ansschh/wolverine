"""Locked-prediction sealed-case orchestrator for Rasyn-Retro v1 (RETRO_PLAN R-7).

Runs the trained system on the 3 sealed cases (RETRO-001 oseltamivir,
RETRO-002 nirmatrelvir, RETRO-003 Rasyn-designed). Locks the top-5
CandidateRoute outputs per case with SHA256 hashes BEFORE any reveal.

For RETRO-003 (no literature target until populated): if
target_canonical_smiles is None in the registry, this script reads the
candidate from a separate JSON `--retro-003-target` so the registry can
be updated after the R-2 ranking step.

Outputs:
    artifacts/retro_stage5_results/
        retro_001/
            candidates.json
            candidates.sha256
            metadata.json
        retro_002/
            ...
        retro_003/
            ...
        manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

from rasyn.synth.retro.orchestrator import (
    CheckpointPaths, OrchestratorConfig, RetroOrchestrator,
)
from rasyn.synth.retro.planner import PlannerConfig
from rasyn.synth.retro.registry import load_retro_sealed_case_registry

logger = logging.getLogger("retro.sealed")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _build_orchestrator(args: argparse.Namespace) -> RetroOrchestrator:
    ckpt = CheckpointPaths(
        template=args.template_ckpt, templates_pickle=args.templates,
        graphedit=args.graphedit_ckpt, seq2seq=args.seq2seq_ckpt,
        retrieval_index=args.retrieval_index, retrieval_metadata=args.retrieval_metadata,
        diffusion=args.diffusion_ckpt, forward=args.forward_ckpt,
        conditions=args.conditions_ckpt, value=args.value_ckpt,
        buyables_parquet=args.buyables,
    )
    return RetroOrchestrator(OrchestratorConfig(
        planner=PlannerConfig(
            max_steps=args.max_steps,
            time_budget_s=args.time_budget_s,
            top_k_routes=args.top_k_routes,
        ),
        ckpt=ckpt,
    ))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("artifacts/retro_stage5_results"))
    p.add_argument("--retro-003-target", type=Path, default=None,
                   help="JSON {smiles, ikey, name} for the Rasyn-designed target")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--time-budget-s", type=float, default=300.0)
    p.add_argument("--top-k-routes", type=int, default=5)
    # checkpoints
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
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    reg = load_retro_sealed_case_registry()
    orch = _build_orchestrator(args)

    manifest = {"runs": [], "system_metadata": {}}
    for case in reg.cases:
        case_out = args.out / case.case_id.lower().replace("-", "_")
        case_out.mkdir(parents=True, exist_ok=True)

        # Resolve target SMILES + InChIKey
        target_smi = case.target_canonical_smiles
        target_ik = case.target_inchi_key
        if not target_smi and case.case_id == "RETRO-003" and args.retro_003_target:
            ref = json.loads(args.retro_003_target.read_text())
            target_smi = ref["smiles"]; target_ik = ref["ikey"]
        if not target_smi or not target_ik:
            logger.warning("skipping %s: no target SMILES", case.case_id)
            continue

        logger.info("running %s: target=%s", case.case_id, target_smi)
        t0 = time.time()
        candidates = orch.plan_routes(target_smiles=target_smi, target_inchi_key=target_ik)
        elapsed = time.time() - t0

        cands_serialised = [c.model_dump() for c in candidates[: args.top_k_routes]]
        payload = json.dumps({
            "case_id": case.case_id,
            "target_smiles": target_smi,
            "target_inchi_key": target_ik,
            "candidates": cands_serialised,
        }, indent=2, sort_keys=True).encode("utf-8")

        sha = _sha256(payload)
        (case_out / "candidates.json").write_bytes(payload)
        (case_out / "candidates.sha256").write_text(sha + "\n")
        (case_out / "metadata.json").write_text(json.dumps({
            "case_id": case.case_id,
            "target_name": case.target_name,
            "n_candidates": len(cands_serialised),
            "elapsed_s": elapsed,
            "sha256": sha,
            "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
        manifest["runs"].append({
            "case_id": case.case_id, "sha256": sha,
            "n_candidates": len(cands_serialised), "elapsed_s": elapsed,
        })
        logger.info("  %s: %d candidates, sha256=%s", case.case_id, len(cands_serialised), sha)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s", args.out / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
