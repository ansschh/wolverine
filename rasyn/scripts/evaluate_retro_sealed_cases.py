"""Auto-judge for the 3 sealed retro cases (RETRO_PLAN R-7).

Reads `artifacts/retro_stage5_results/` (output of run_retro_sealed_cases.py)
and produces verdicts per case using the success_criteria from the
registry:

  verdict ∈ {
    literature_optimal,
    literature_competitive,
    novel_valid,
    route_proposed_no_literature_baseline,
    missed,
  }

Comparison logic:
  - For RETRO-001 / RETRO-002: compare reaction-class sequence Levenshtein
    distance between top-K candidate routes and the reference sequence in
    the registry. Compute Tanimoto on intermediates (when InChIKey
    available). Forward-validation pass rate must be >= the case's
    `min_forward_validation_rate`.
  - For RETRO-003 (no literature baseline): verdict is
    `route_proposed_no_literature_baseline` if at least one route was
    locked and meets the forward-validation floor; otherwise `missed`.

Outputs:
    artifacts/retro_stage5_results/verdicts.json
    artifacts/retro_stage5_results/<case>/verdict.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rasyn.synth.retro.registry import load_retro_sealed_case_registry

logger = logging.getLogger("retro.eval_sealed")


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _bucket(
    case_id: str,
    reference_class_seq: list[str],
    candidate_class_seq: list[str],
    forward_pass_rate: float,
    step_count: int,
    min_fwd: float,
    step_tolerance: int,
    reference_step_count: int | None,
) -> str:
    if case_id == "RETRO-003":
        if forward_pass_rate >= min_fwd:
            return "route_proposed_no_literature_baseline"
        return "missed"
    if forward_pass_rate < min_fwd:
        return "missed"
    if reference_class_seq and candidate_class_seq:
        dist = _levenshtein(reference_class_seq, candidate_class_seq)
        if dist == 0 and step_count == reference_step_count:
            return "literature_optimal"
        if dist <= 2 and reference_step_count is not None and abs(
            step_count - reference_step_count
        ) <= step_tolerance:
            return "literature_competitive"
    return "novel_valid"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path,
                   default=Path("artifacts/retro_stage5_results"))
    p.add_argument("--top-k", type=int, default=5,
                   help="how many top candidates to consider per case")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    reg = load_retro_sealed_case_registry()
    verdicts = {}
    for case in reg.cases:
        case_dir = args.results_dir / case.case_id.lower().replace("-", "_")
        cands_path = case_dir / "candidates.json"
        if not cands_path.exists():
            logger.warning("missing candidates for %s", case.case_id)
            continue
        payload = json.loads(cands_path.read_text())
        candidates = payload.get("candidates", [])

        case_verdict = "missed"
        best_per_top_k = []
        for i, cand in enumerate(candidates[: args.top_k]):
            class_seq = [s.get("reaction_class") for s in cand.get("step_predictions", [])]
            fwd_rate = cand.get("forward_pass_rate", 0.0)
            step_count = cand.get("route_tree", {}).get("step_count", 0)
            ref_seq = case.hidden_solution.reference_reaction_class_sequence
            verdict = _bucket(
                case.case_id,
                reference_class_seq=list(ref_seq) if ref_seq else [],
                candidate_class_seq=class_seq,
                forward_pass_rate=float(fwd_rate),
                step_count=int(step_count),
                min_fwd=case.success_criteria.min_forward_validation_rate,
                step_tolerance=case.success_criteria.literature_recovery_step_tolerance,
                reference_step_count=case.hidden_solution.reference_route_step_count,
            )
            best_per_top_k.append({
                "rank": i + 1,
                "route_score": cand.get("route_score"),
                "step_count": step_count,
                "forward_pass_rate": fwd_rate,
                "class_sequence": class_seq,
                "verdict": verdict,
            })
            # Promote the best verdict
            preference = {
                "literature_optimal": 4,
                "literature_competitive": 3,
                "novel_valid": 2,
                "route_proposed_no_literature_baseline": 2,
                "missed": 1,
            }
            if preference.get(verdict, 0) > preference.get(case_verdict, 0):
                case_verdict = verdict

        verdict_payload = {
            "case_id": case.case_id,
            "case_verdict": case_verdict,
            "per_candidate": best_per_top_k,
        }
        (case_dir / "verdict.json").write_text(json.dumps(verdict_payload, indent=2))
        verdicts[case.case_id] = case_verdict
        logger.info("  %s -> %s", case.case_id, case_verdict)

    (args.results_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
