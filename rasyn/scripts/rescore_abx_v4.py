"""Round-1 post-hoc rescorer: apply Round-1 fixes to v3 predictions WITHOUT retraining.

Three changes to the composite scoring:

  1. **Multiplicative composite** instead of additive subtraction:
       final = ab * (1 - cyto)^alpha * (1 - artifact)^beta
     ensures a cytotoxic-active (high ab AND high cyto) cannot win.

  2. **Novelty bonus** — reward candidates with low Tanimoto to the hidden
     answer's training-active neighborhood (we approximate this by the
     `max_tanimoto_to_organism_active` field already in the v3 parquet).
     A candidate that's NEW relative to the training-active manifold gets
     a bonus.

  3. **Anti-memorization** — penalize candidates whose
     `max_tanimoto_to_organism_active` is too high (Tan > 0.95), because
     those are training-set replicas (no discovery).

Reads the v3 per-case parquets, recomputes final_discovery_score, re-ranks,
and writes the new top-K + closed-ranking metrics + summary card.

Usage:
    python scripts/rescore_abx_v4.py \\
        --in-dir  artifacts/abx_stage5_v3 \\
        --out-dir artifacts/abx_stage5_v4_rescore \\
        --registry rasyn/antibiotic/sealed_case_registry.yaml \\
        --alpha 2.0 --beta 1.0 \\
        --novelty-weight 0.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def rescore_v4(df: pd.DataFrame, alpha: float, beta: float,
                novelty_weight: float, memorization_threshold: float) -> pd.DataFrame:
    df = df.copy()
    ab = df["antibacterial_score"].astype(float).clip(0, 1)
    cy = df["cytotox_risk"].astype(float).clip(0, 1)
    ar = df["artifact_risk"].astype(float).clip(0, 1)

    # Multiplicative selectivity: ab * (1 - cyto)^alpha * (1 - artifact)^beta
    selectivity_factor = (1 - cy).pow(alpha) * (1 - ar).pow(beta)

    # Novelty bonus: reward LOW similarity to organism-active neighborhood.
    # If `max_tanimoto_to_organism_active` exists, novelty = 1 - it. Else 0.5 default.
    if "max_tanimoto_to_organism_active" in df.columns:
        tan_to_active = df["max_tanimoto_to_organism_active"].fillna(0.0).astype(float).clip(0, 1)
    else:
        tan_to_active = pd.Series([0.0] * len(df))
    novelty = 1.0 - tan_to_active

    # Anti-memorization: zero out anyone with Tan > memorization_threshold
    # (they're effectively training-set replicas — not discovery)
    memorization_mask = tan_to_active >= memorization_threshold
    novelty_clipped = novelty.where(~memorization_mask, 0.0)

    base = ab * selectivity_factor
    df["v4_selectivity_factor"] = selectivity_factor
    df["v4_novelty"] = novelty_clipped
    df["v4_is_memorization"] = memorization_mask
    df["v4_base_score"] = base
    df["v4_final_score"] = base * (1.0 + novelty_weight * novelty_clipped)
    return df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--registry", type=Path,
                   default=Path("rasyn/antibiotic/sealed_case_registry.yaml"))
    p.add_argument("--alpha", type=float, default=2.0,
                   help="Cytotox exponent (higher = more penalty)")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Artifact exponent")
    p.add_argument("--novelty-weight", type=float, default=0.3)
    p.add_argument("--memorization-threshold", type=float, default=0.95,
                   help="Tan-to-organism-active above which we treat as training replica")
    p.add_argument("--cases", default="ABX-001,ABX-002,ABX-003")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reg = yaml.safe_load(args.registry.read_text())
    cases_by_id = {c["case_id"]: c for c in reg["cases"]}

    summary: dict = {}
    for case_id in args.cases.split(","):
        case_id = case_id.strip()
        in_pq = args.in_dir / f"{case_id}_top_candidates.parquet"
        if not in_pq.exists():
            print(f"  {case_id} missing input parquet — skipping")
            continue
        df = pd.read_parquet(in_pq)
        df_re = rescore_v4(df, args.alpha, args.beta,
                            args.novelty_weight, args.memorization_threshold)
        df_re = df_re.sort_values("v4_final_score", ascending=False).reset_index(drop=True)
        df_re["v4_rank"] = df_re.index + 1

        # Find hidden hit rank
        ans_smi = (cases_by_id.get(case_id, {}).get("hidden_solution") or {}).get("canonical_smiles")
        hidden_rank = None
        if ans_smi:
            matches = df_re[df_re["candidate_smiles"] == ans_smi]
            if not matches.empty:
                hidden_rank = int(matches["v4_rank"].iloc[0])

        # Original rank for comparison
        df_orig = df.sort_values("final_discovery_score", ascending=False).reset_index(drop=True)
        df_orig["orig_rank"] = df_orig.index + 1
        orig_rank = None
        if ans_smi:
            matches = df_orig[df_orig["candidate_smiles"] == ans_smi]
            if not matches.empty:
                orig_rank = int(matches["orig_rank"].iloc[0])

        # Top-20 v4
        top20 = df_re.head(20)[[
            "v4_rank", "v4_final_score", "v4_base_score", "v4_selectivity_factor",
            "v4_novelty", "v4_is_memorization",
            "antibacterial_score", "cytotox_risk", "artifact_risk", "channel", "candidate_smiles",
        ]]
        top20.to_parquet(args.out_dir / f"{case_id}_v4_top20.parquet", index=False)

        # Card
        lines = [f"# {case_id} — v4 rescore (multiplicative + novelty + anti-memorization)", ""]
        lines.append(f"**Hidden answer SMILES:** `{ans_smi or 'n/a'}`")
        lines.append(f"**Library size:** {len(df_re)}")
        lines.append(f"**v3 rank (additive composite):** {orig_rank}")
        lines.append(f"**v4 rank (multiplicative + novelty):** {hidden_rank}")
        if hidden_rank is not None and orig_rank is not None:
            delta = hidden_rank - orig_rank
            lines.append(f"**Δ rank:** {delta:+d} ({'better' if delta < 0 else 'worse' if delta > 0 else 'same'})")
        lines.append("")
        lines.append("## Top-20 v4 ranked")
        lines.append("")
        lines.append("| v4 rank | final | base | selectivity | novelty | memo? | ab | cyto | art | channel | SMILES |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in top20.iterrows():
            lines.append(
                f"| {int(r['v4_rank'])} | {r['v4_final_score']:.4f} | {r['v4_base_score']:.4f} | "
                f"{r['v4_selectivity_factor']:.3f} | {r['v4_novelty']:.2f} | {'Y' if r['v4_is_memorization'] else 'n'} | "
                f"{r['antibacterial_score']:.3f} | {r['cytotox_risk']:.3f} | {r['artifact_risk']:.3f} | "
                f"{r.get('channel', '?')} | `{r['candidate_smiles'][:60]}` |"
            )
        (args.out_dir / f"{case_id}_v4_card.md").write_text("\n".join(lines), encoding="utf-8")

        summary[case_id] = {
            "orig_rank": orig_rank,
            "v4_rank": hidden_rank,
            "library_size": len(df_re),
            "delta": (hidden_rank - orig_rank) if (hidden_rank and orig_rank) else None,
        }
        print(f"  {case_id}: orig_rank={orig_rank} v4_rank={hidden_rank} "
              f"delta={summary[case_id]['delta']}  lib={len(df_re)}")

    (args.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out_dir}/_summary.json")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
