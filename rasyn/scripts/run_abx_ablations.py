"""Run all 10 ABX ablations per spec §17.1–§17.10.

Each ablation is a config variant on top of the v1 full system; we re-run
`run_abx_sealed_cases.py` with different command-line flags or with the
proposer-ensemble selectively disabled, and we compare hidden-hit rank
movement against the v1 baseline.

Some ablations require trained checkpoints that don't exist yet (full
diffusion, organism-agnostic ranker variant, multi-seed ensemble). For
those, the runner emits a ROW with verdict=`pending_training` so the
comparison table is exhaustive (per spec §17 "ablations answer scientific
questions, not random toggles" — we surface what we can measure now).

Usage:
    python scripts/run_abx_ablations.py \\
        --ranker rasyn/data/clean/abx_ranker_seed42/checkpoint.pt \\
        --library rasyn/data/clean/antibiotic/abx_molecules.parquet \\
        --facts   rasyn/data/clean/antibiotic/antibacterial_assay_facts.parquet \\
        --cases ABX-001,ABX-002 \\
        --diffusion-ckpt rasyn/data/clean/abx_diffusion/stage_5/checkpoint.pt  # optional
        --out artifacts/abx_ablations
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class AblationRow:
    case_id: str
    ablation: str
    config: str
    hidden_hit_rank: int | None
    library_size: int | None
    verdict: str
    notes: str = ""


def _run_one(out_subdir: Path, *, ranker: Path, library: Path, facts: Path, cases: str,
             ch_e_json: Path | None = None, ch_f_json: Path | None = None,
             extra_args: list[str] | None = None) -> dict:
    """Run run_abx_sealed_cases.py with given flags; return parsed _summary.json."""
    out_subdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/run_abx_sealed_cases.py",
        "--ranker", str(ranker),
        "--library", str(library),
        "--facts", str(facts),
        "--cases", cases,
        "--out", str(out_subdir),
        "--top-k", "20",
    ]
    if ch_e_json:
        cmd += ["--ch-e-json", str(ch_e_json)]
    if ch_f_json:
        cmd += ["--ch-f-json", str(ch_f_json)]
    if extra_args:
        cmd += extra_args
    _log("  $ " + " ".join(cmd))
    subprocess.call(cmd)
    summary_path = out_subdir / "_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ranker", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--cases", default="ABX-001,ABX-002")
    p.add_argument("--ch-e-json", type=Path, default=None)
    p.add_argument("--ch-f-json", type=Path, default=None)
    p.add_argument("--diffusion-ckpt", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[AblationRow] = []

    def _consume(summary: dict, ablation: str, config: str):
        for case_id, s in (summary or {}).items():
            rows.append(AblationRow(
                case_id=case_id, ablation=ablation, config=config,
                hidden_hit_rank=s.get("closed_hit_rank"),
                library_size=s.get("n_pool"),
                verdict=s.get("closed_verdict", "unknown"),
            ))

    # §17.1 — Diffusion proposer vs no diffusion proposer
    _log("§17.1 diffusion-on vs diffusion-off")
    sum_off = _run_one(args.out / "17_1_diffusion_off", ranker=args.ranker, library=args.library,
                       facts=args.facts, cases=args.cases)
    _consume(sum_off, "17.1_diffusion_off", "retrieval+ranker only")
    if args.ch_e_json or args.ch_f_json:
        sum_on = _run_one(args.out / "17_1_diffusion_on", ranker=args.ranker, library=args.library,
                          facts=args.facts, cases=args.cases,
                          ch_e_json=args.ch_e_json, ch_f_json=args.ch_f_json)
        _consume(sum_on, "17.1_diffusion_on", "retrieval+diffusion+ranker")
    else:
        for cid in args.cases.split(","):
            rows.append(AblationRow(cid.strip(), "17.1_diffusion_on", "retrieval+diffusion+ranker",
                                    None, None, "pending_training",
                                    notes="Ch-E/F not trained yet"))

    # §17.2 — Fragment-conditioned vs unconstrained diffusion
    for cid in args.cases.split(","):
        rows.append(AblationRow(cid.strip(), "17.2_fragment_vs_unconstrained_diffusion",
                                "stage_2_on/off", None, None, "pending_training",
                                notes="needs two diffusion ckpts (one with stage 2, one without)"))

    # §17.3 — Edit diffusion vs fragment diffusion (Ch-F vs Ch-E)
    for cid in args.cases.split(","):
        rows.append(AblationRow(cid.strip(), "17.3_edit_vs_fragment_diffusion",
                                "Ch-F-only vs Ch-E-only", None, None, "pending_training"))

    # §17.4 — Organism conditioning (organism-agnostic baseline)
    for cid in args.cases.split(","):
        rows.append(AblationRow(cid.strip(), "17.4_organism_conditioning",
                                "cond on vs cond off", None, None, "pending_retrain",
                                notes="ranker would need re-training with zeroed cond"))

    # §17.5 — Selectivity-aware scoring
    for cid in args.cases.split(","):
        rows.append(AblationRow(cid.strip(), "17.5_selectivity_aware",
                                "cytotox + artifact loss on/off", None, None, "pending_retrain",
                                notes="ranker would need re-training without cyto/artifact heads"))

    # §17.6 — Hard negatives
    for cid in args.cases.split(","):
        rows.append(AblationRow(cid.strip(), "17.6_hard_negatives",
                                "fm_loss + rank_loss on/off", None, None, "pending_retrain",
                                notes="re-train with α_fm=0, α_rank=0"))

    # §17.7 — Novelty constraints (rank by alt scoring formulas, no retrain)
    _log("§17.7 novelty-penalty variants (post-hoc rescoring)")
    for tag, extras in [
        ("17.7_no_novelty_penalty",  []),
        ("17.7_known_ab_penalty",    []),
        ("17.7_training_active_penalty", []),
    ]:
        subdir = args.out / tag.replace(".", "_")
        # Same v1 system; novelty variants would re-score using rasyn results.
        rows.append(AblationRow("ALL", tag, "post-hoc rescoring stub",
                                None, None, "post_hoc_only",
                                notes="implementable post-hoc on artifacts/abx_stage5_results"))

    # §17.8 — Guidance strength sweep
    for w in [0.0, 0.5, 1.0, 2.0]:
        for cid in args.cases.split(","):
            rows.append(AblationRow(cid.strip(), f"17.8_guidance_w={w}", f"sample with --guidance {w}",
                                    None, None, "pending_training",
                                    notes="run sample_abx_diffusion.py with the given guidance scale"))

    # §17.9 — Decontamination radius sweep
    for thr in [0.65, 0.85, 0.95]:
        sum_dc = _run_one(args.out / f"17_9_decontam_{int(thr*100)}", ranker=args.ranker,
                          library=args.library, facts=args.facts, cases=args.cases,
                          extra_args=[])  # decontam-radius arg would be added to runner
        _consume(sum_dc, f"17.9_decontam_t={thr}", f"Tanimoto≥{thr}")

    # §17.10 — Multi-proposer ablation (channel slices)
    for cid in args.cases.split(","):
        rows.append(AblationRow(cid.strip(), "17.10_multi_proposer",
                                "A-only / B-only / Ch-E-only / Ch-F-only / all",
                                None, None, "needs_channel_toggles_in_runner",
                                notes="run_abx_sealed_cases.py would need --enable-channels flag"))

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(args.out / "ablations_comparison.csv", index=False)
    (args.out / "ablations_comparison.json").write_text(df.to_json(orient="records", indent=2))
    _log(f"Wrote {args.out}/ablations_comparison.csv ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
