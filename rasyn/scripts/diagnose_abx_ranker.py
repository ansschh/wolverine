"""Forensic diagnostic of the ABX ranker.

Answers: WHY does the v3 ranker put halicin at rank 1091/1130 (96.5th
percentile from the top) instead of top-100?

This script logs **everything humanly inspectable** about the inputs,
intermediates, and outputs for the hidden hits + top-ranked candidates.
Designed to run on the training pod (data + ckpts are local there) and
dump a single JSON blob that you can pull back and read by hand.

What gets logged per (case, target_molecule):
  1. Raw SMILES + InChIKey + n_atoms + RDKit descriptors
  2. Tokenization: list of (char, token_id) tuples, sequence length, OOV chars
  3. Morgan-FP fingerprint + Murcko scaffold SMILES
  4. Top-K Tanimoto neighbors in the ABX_FACTS table, with their organism +
     activity_label.  THIS is where we see whether halicin had ANY training
     analogs labeled 'active' against E.coli.
  5. Per-seed ranker output: all 12 heads (antibacterial, cytotox,
     organism_specific, selectivity, hemolysis, artifact, novelty,
     synthesizability, uncertainty, known_ab_pen, training_active_pen,
     failure_mode_probs) for each of seeds 42/43/44.
  6. Composite final_discovery_score and ensemble std.
  7. Same logged for the top-1 ranker candidate so we can diff "what the
     ranker likes" against "what the answer looks like".

What gets logged per case (global stats):
  - antibacterial_score distribution across the full pool (mean, p1, p50, p99)
  - Where the hidden hit sits in that distribution (rank + percentile)
  - Training-set "active" count per organism
  - Training-set "active" rows within Tan >= 0.3 of the hidden hit

Run on pod:
    cd /workspace/wolverine/rasyn
    python scripts/diagnose_abx_ranker.py \\
        --ranker rasyn/data/clean/abx_ranker_seed42/checkpoint.pt,rasyn/data/clean/abx_ranker_seed43/checkpoint.pt,rasyn/data/clean/abx_ranker_seed44/checkpoint.pt \\
        --library rasyn/data/clean/antibiotic/abx_molecules.parquet \\
        --facts   rasyn/data/clean/antibiotic/antibacterial_assay_facts.parquet \\
        --registry rasyn/antibiotic/sealed_case_registry.yaml \\
        --top-k-neighbors 25 \\
        --top-k-candidates 5 \\
        --out /tmp/abx_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from train_abx_ranker import (  # noqa: E402
    ABXMultiHeadRanker, condition_vector, ORGANISM_LIST, GRAM_LIST, SPECTRUM_LIST,
    FAILURE_MODES, tokenize as _tok,
)
from h200_smiles_lm_pretrain import VOCAB, VOCAB_SIZE  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rdkit_descriptors(smi: str) -> dict:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return {"valid": False}
    return {
        "valid": True,
        "n_atoms": m.GetNumAtoms(),
        "n_heavy_atoms": m.GetNumHeavyAtoms(),
        "n_rings": rdMolDescriptors.CalcNumRings(m),
        "n_aromatic_rings": Lipinski.NumAromaticRings(m),
        "molecular_weight": float(Descriptors.MolWt(m)),
        "clogp": float(Descriptors.MolLogP(m)),
        "tpsa": float(Descriptors.TPSA(m)),
        "hba": int(Lipinski.NumHAcceptors(m)),
        "hbd": int(Lipinski.NumHDonors(m)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(m)),
        "fsp3": float(Lipinski.FractionCSP3(m)),
        "formal_charge": int(Chem.rdmolops.GetFormalCharge(m)),
        "canonical_smiles": Chem.MolToSmiles(m, canonical=True),
        "inchi_key": Chem.MolToInchiKey(m),
    }


def _murcko_smiles(smi: str) -> str | None:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    sf = GetScaffoldForMol(m)
    return Chem.MolToSmiles(sf, canonical=True) if sf else None


def _morgan_fp(smi: str, n_bits: int = 2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits) if m else None


def _tokenize_diagnostic(smi: str, max_len: int = 128) -> dict:
    """Return character-by-character tokenization, OOV detection."""
    UNK = 1
    PAD = VOCAB.get("[PAD]", 0)
    chars = list(smi)
    token_ids = [VOCAB.get(c, UNK) for c in chars]
    oov_chars = sorted({c for c, tid in zip(chars, token_ids) if tid == UNK})
    return {
        "raw_smiles": smi,
        "n_chars": len(chars),
        "truncated_to": min(len(chars), max_len),
        "would_truncate": len(chars) > max_len,
        "tokens_first_40": [(c, int(tid)) for c, tid in zip(chars[:40], token_ids[:40])],
        "oov_chars": oov_chars,
        "n_oov": len(oov_chars),
    }


def _ranker_forward(models: list, smi: str, organism: str, gram: str, spectrum: str,
                     device, max_len: int = 128) -> dict:
    """Run all 3 ranker ckpts on the SMILES; return all 12 head outputs per ckpt."""
    ids, mask = _tok(smi or "C", max_len)
    ids_t = torch.from_numpy(ids).long().unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
    cond = torch.from_numpy(condition_vector(organism, gram, spectrum)).unsqueeze(0).to(device)
    per_seed = []
    for i, m in enumerate(models):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = m(ids_t, mask_t, cond)
        seed_out = {}
        for head_name, tensor in out.items():
            if tensor.dim() == 2 and tensor.size(-1) == len(FAILURE_MODES):
                # failure_modes is a 5-way softmax
                seed_out[head_name] = dict(zip(FAILURE_MODES,
                                                tensor.float().softmax(-1).cpu().tolist()[0]))
            else:
                seed_out[head_name] = float(tensor.float().cpu().tolist()[0]
                                             if tensor.dim() <= 1 or tensor.size(-1) == 1
                                             else tensor.float().cpu().tolist()[0][0])
        per_seed.append(seed_out)
    # Ensemble mean + std
    head_names = list(per_seed[0].keys())
    ens = {}
    for h in head_names:
        vals = [s[h] for s in per_seed]
        if isinstance(vals[0], dict):
            mean = {k: sum(v[k] for v in vals) / len(vals) for k in vals[0]}
            ens[h] = mean
        else:
            arr = np.array(vals, dtype=float)
            ens[h] = {"mean": float(arr.mean()), "std": float(arr.std()),
                      "per_seed": arr.tolist()}
    return {"per_seed_full": per_seed, "ensemble": ens}


def _topk_neighbors(target_smi: str, library_df: pd.DataFrame,
                     facts_df: pd.DataFrame | None, top_k: int = 20) -> list[dict]:
    """Find top-K Tanimoto neighbors of target_smi in library_df. Return rows
    annotated with their `activity_label` + `organism` if facts_df is provided."""
    from rdkit.DataStructs import TanimotoSimilarity
    target_fp = _morgan_fp(target_smi)
    if target_fp is None:
        return []
    lib_smiles = library_df["canonical_smiles"].dropna().astype(str).tolist()
    sims = []
    for i, s in enumerate(lib_smiles):
        if i % 10000 == 0:
            _log(f"  scoring neighbor {i}/{len(lib_smiles)}")
        fp = _morgan_fp(s)
        if fp is None:
            continue
        sims.append((s, TanimotoSimilarity(target_fp, fp)))
    sims.sort(key=lambda x: -x[1])
    out: list[dict] = []
    for smi, tan in sims[:top_k]:
        row: dict = {"smiles": smi, "tanimoto": float(tan)}
        if facts_df is not None:
            matches = facts_df[facts_df["canonical_smiles"] == smi]
            if not matches.empty:
                # most-confident activity-label across rows
                if "activity_label" in matches.columns:
                    row["activity_labels"] = matches["activity_label"].value_counts().to_dict()
                if "organism" in matches.columns:
                    row["organisms"] = matches["organism"].value_counts().to_dict()
        out.append(row)
    return out


def _training_active_stats(facts_df: pd.DataFrame, target_smi: str,
                             tan_threshold: float = 0.30) -> dict:
    """How many training 'active' rows are within Tan >= threshold of target?"""
    from rdkit.DataStructs import TanimotoSimilarity
    target_fp = _morgan_fp(target_smi)
    if target_fp is None:
        return {}
    actives = facts_df[facts_df["activity_label"] == "active"]
    actives_unique = actives.drop_duplicates("canonical_smiles")
    n_near = 0
    near_rows: list[dict] = []
    near_by_organism: dict[str, int] = {}
    for _, r in actives_unique.iterrows():
        smi = r.get("canonical_smiles")
        if not smi:
            continue
        fp = _morgan_fp(smi)
        if fp is None:
            continue
        tan = TanimotoSimilarity(target_fp, fp)
        if tan >= tan_threshold:
            n_near += 1
            org = r.get("organism", "unknown")
            near_by_organism[org] = near_by_organism.get(org, 0) + 1
            if len(near_rows) < 25:
                near_rows.append({"smiles": smi, "tanimoto": float(tan),
                                    "organism": org})
    return {
        "n_unique_actives_in_training": int(len(actives_unique)),
        "n_actives_within_tan_threshold": n_near,
        "tan_threshold": tan_threshold,
        "near_actives_by_organism": near_by_organism,
        "top_25_near_active_rows": sorted(near_rows, key=lambda x: -x["tanimoto"])[:25],
    }


def _pool_score_distribution(models, pool_smiles: list[str], organism: str,
                               gram: str, spectrum: str, device, max_len: int = 128,
                               bs: int = 32) -> dict:
    """Run ranker on entire candidate pool; return antibacterial_score distribution."""
    scores: list[float] = []
    cond = torch.from_numpy(condition_vector(organism, gram, spectrum)).to(device)
    for i in range(0, len(pool_smiles), bs):
        chunk = pool_smiles[i:i+bs]
        ids_list, mask_list = [], []
        for s in chunk:
            ids, mask = _tok(s or "C", max_len)
            ids_list.append(ids); mask_list.append(mask)
        ids_t = torch.from_numpy(np.stack(ids_list)).long().to(device)
        mask_t = torch.from_numpy(np.stack(mask_list)).bool().to(device)
        cond_b = cond.unsqueeze(0).expand(len(chunk), -1)
        per_seed_ab = []
        for m in models:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = m(ids_t, mask_t, cond_b)
            per_seed_ab.append(out["antibacterial"].float().cpu().numpy())
        ens_ab = np.mean(np.stack(per_seed_ab), axis=0)
        scores.extend(ens_ab.tolist())
    arr = np.array(scores, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p1": float(np.percentile(arr, 1)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "raw_scores": scores,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ranker", required=True,
                   help="Comma-separated list of ranker checkpoint paths")
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--registry", type=Path,
                   default=Path("rasyn/antibiotic/sealed_case_registry.yaml"))
    p.add_argument("--cases", default="ABX-001,ABX-002,ABX-003")
    p.add_argument("--top-k-neighbors", type=int, default=25)
    p.add_argument("--top-k-candidates", type=int, default=5)
    p.add_argument("--tan-threshold", type=float, default=0.30)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    _log("Loading library + facts...")
    library_df = pd.read_parquet(args.library)
    facts_df = pd.read_parquet(args.facts)
    _log(f"  library {len(library_df):,} mols | facts {len(facts_df):,} rows")

    _log("Loading sealed-case registry...")
    reg = yaml.safe_load(args.registry.read_text())
    cases_by_id = {c["case_id"]: c for c in reg["cases"]}

    _log("Loading ranker ckpts...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_paths = [Path(p.strip()) for p in args.ranker.split(",") if p.strip()]
    models = []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cargs = ckpt.get("args", {})
        m = ABXMultiHeadRanker(
            VOCAB_SIZE,
            d_model=cargs.get("d_model", 1024),
            n_heads=cargs.get("n_heads", 16),
            n_layers=cargs.get("n_layers", 16),
            max_len=cargs.get("max_len", 128),
        ).to(device)
        sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
        m.load_state_dict(sd, strict=True)
        m.eval()
        models.append(m)
        _log(f"  loaded {ckpt_path.name}")

    # Training-distribution global stats
    _log("Computing training distribution stats...")
    active_per_org = (facts_df[facts_df["activity_label"] == "active"]
                      .drop_duplicates("canonical_smiles")
                      .groupby("organism").size().to_dict())
    inactive_per_org = (facts_df[facts_df["activity_label"] == "inactive"]
                        .drop_duplicates("canonical_smiles")
                        .groupby("organism").size().to_dict())
    diagnostic: dict = {
        "training_distribution": {
            "n_facts_rows": int(len(facts_df)),
            "n_unique_molecules": int(facts_df["canonical_smiles"].nunique()),
            "activity_label_counts": facts_df["activity_label"].value_counts().to_dict(),
            "n_unique_actives_per_organism": {k: int(v) for k, v in active_per_org.items()},
            "n_unique_inactives_per_organism": {k: int(v) for k, v in inactive_per_org.items()},
        },
        "cases": {},
    }

    pool_smiles = library_df["canonical_smiles"].dropna().astype(str).tolist()

    for case_id in args.cases.split(","):
        case_id = case_id.strip()
        case = cases_by_id.get(case_id)
        if case is None:
            continue
        ans_smi = (case.get("hidden_solution") or {}).get("canonical_smiles")
        if not ans_smi:
            _log(f"[{case_id}] no hidden SMILES — skipping diagnostic")
            diagnostic["cases"][case_id] = {"skipped": "no_hidden_smiles"}
            continue
        org_ctx = case.get("organism_context") or {}
        organism = org_ctx.get("organism", "unknown")
        gram = org_ctx.get("gram_type", "unknown")
        spectrum = org_ctx.get("spectrum_goal", "unknown")

        _log(f"\n===== {case_id} =====")
        _log(f"  organism={organism} gram={gram} spectrum={spectrum}")
        _log(f"  hidden answer SMILES: {ans_smi}")

        case_diag: dict = {
            "organism_context": {"organism": organism, "gram_type": gram,
                                    "spectrum_goal": spectrum},
        }

        # ------- HIDDEN ANSWER analysis -------
        _log(f"  [hidden answer] rdkit descriptors + tokenization")
        hidden_descriptors = _rdkit_descriptors(ans_smi)
        hidden_tokens = _tokenize_diagnostic(ans_smi)
        hidden_murcko = _murcko_smiles(ans_smi)

        _log(f"  [hidden answer] running through 3 ranker seeds")
        hidden_ranker_out = _ranker_forward(models, ans_smi, organism, gram, spectrum, device)

        _log(f"  [hidden answer] finding top-{args.top_k_neighbors} Tanimoto neighbors in library")
        hidden_neighbors_lib = _topk_neighbors(ans_smi, library_df, facts_df,
                                                top_k=args.top_k_neighbors)

        _log(f"  [hidden answer] counting training-active analogs (Tan >= {args.tan_threshold})")
        hidden_training_actives = _training_active_stats(facts_df, ans_smi,
                                                          tan_threshold=args.tan_threshold)

        # Composite final score (matches run_abx_sealed_cases formula)
        ab = hidden_ranker_out["ensemble"]["antibacterial"]["mean"]
        cy = hidden_ranker_out["ensemble"]["cytotox"]["mean"]
        ar = hidden_ranker_out["ensemble"]["artifact"]["mean"]
        hidden_composite = ab - 0.5 * cy - 0.3 * ar

        case_diag["hidden_answer"] = {
            "raw_smiles": ans_smi,
            "rdkit_descriptors": hidden_descriptors,
            "murcko_scaffold_smiles": hidden_murcko,
            "tokenization": hidden_tokens,
            "ranker_outputs": hidden_ranker_out,
            "composite_final_discovery_score": float(hidden_composite),
            "top_neighbors_in_library": hidden_neighbors_lib,
            "training_active_neighborhood": hidden_training_actives,
        }

        # ------- POOL DISTRIBUTION -------
        _log(f"  [pool] scoring full library ({len(pool_smiles):,}) for antibacterial_score distribution")
        pool_dist = _pool_score_distribution(models, pool_smiles, organism, gram, spectrum,
                                              device)
        # Where does the hidden hit fall?
        hit_ab = ab
        n_above = sum(1 for s in pool_dist["raw_scores"] if s > hit_ab)
        hidden_pool_rank = n_above + 1  # 1-indexed
        pool_dist["hidden_hit_antibacterial_score"] = float(hit_ab)
        pool_dist["hidden_hit_rank_by_antibacterial_score"] = int(hidden_pool_rank)
        pool_dist["hidden_hit_percentile_from_top"] = (
            float(hidden_pool_rank) / max(1, pool_dist["n"]) * 100.0
        )
        # don't save the raw scores list to keep JSON small
        pool_dist_serializable = {k: v for k, v in pool_dist.items() if k != "raw_scores"}
        case_diag["pool_score_distribution"] = pool_dist_serializable

        # ------- TOP-K CANDIDATES analysis -------
        # Find top-K by antibacterial_score
        raw = pool_dist["raw_scores"]
        top_idx = np.argsort(raw)[-args.top_k_candidates:][::-1]
        top_candidates = []
        for rank_i, idx in enumerate(top_idx, start=1):
            cand_smi = pool_smiles[idx]
            _log(f"  [top-{rank_i}] {cand_smi[:80]} (ab_score={raw[idx]:.3f})")
            cand_desc = _rdkit_descriptors(cand_smi)
            cand_ranker = _ranker_forward(models, cand_smi, organism, gram, spectrum, device)
            cand_neighbors = _topk_neighbors(cand_smi, library_df, facts_df, top_k=5)
            top_candidates.append({
                "rank": rank_i,
                "smiles": cand_smi,
                "descriptors": cand_desc,
                "ranker_outputs": cand_ranker,
                "top_5_lib_neighbors": cand_neighbors,
            })
        case_diag["top_ranked_candidates"] = top_candidates

        # ------- WHAT DOES THE RANKER SEE DIFFERENTLY? -------
        # Diff: hidden_hit's ranker output vs top-1's ranker output
        if top_candidates:
            top1_ens = top_candidates[0]["ranker_outputs"]["ensemble"]
            hidden_ens = hidden_ranker_out["ensemble"]
            # only diff scalar heads
            diff = {}
            for head, val in top1_ens.items():
                if isinstance(val, dict) and "mean" in val:
                    diff[head] = {
                        "top1_mean": val["mean"],
                        "hidden_mean": hidden_ens[head]["mean"],
                        "delta_top1_minus_hidden": val["mean"] - hidden_ens[head]["mean"],
                    }
            case_diag["delta_top1_vs_hidden"] = diff

        diagnostic["cases"][case_id] = case_diag

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(diagnostic, indent=2, default=str))
    _log(f"\nDiagnostic written -> {args.out}")
    _log(f"  Output JSON size: {args.out.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
