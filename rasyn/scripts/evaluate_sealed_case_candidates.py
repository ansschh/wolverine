"""Objective evaluation of Stage-5 top-K candidates vs the literature answer.

For each sealed case:
  1. Load top-K candidates from stage5_results.../{case_id}_top_candidates.parquet
  2. Append the literature answer SMILES as a separate row
  3. Run aux ADMET predictor (aux_finetuned_frozen) on the combined set
  4. Compute RDKit physicochemical descriptors (logP, TPSA, QED, SAScore, alert counts)
  5. Compute Tanimoto-to-parent (preservation proxy)
  6. Compute liability-specific composite "rescue fitness"
  7. Rank everything; show where literature answer lands

Output: side-by-side table per case + JSON with raw metrics.

Hypothesis under test: "is the ranker's top pick objectively better than the
literature answer on predicted ADMET + descriptors?"

Run on Pod A (has Stage-2 + aux ckpts + RDKit):
    cd ~/wolverine/rasyn && source .venv/bin/activate
    python scripts/evaluate_sealed_case_candidates.py \\
        --aux-ckpt rasyn/data/clean/aux_finetuned_frozen/checkpoint.pt \\
        --cases ADMET-001,ADMET-002,ADMET-003 \\
        --out rasyn/data/clean/sealed_case_evaluation
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

from h200_train_aux_admet import (  # type: ignore
    MultiTaskADMET, VOCAB, VOCAB_SIZE, PAD, TDC_DATASETS,
)


PARENT_SMI = {
    "ADMET-001": ("terfenadine",  "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1"),
    "ADMET-002": ("acyclovir",    "Nc1nc(=O)c2ncn(COCCO)c2[nH]1"),
    "ADMET-003": ("OXS007570",    "Cc1cc(F)ccc1-c1cc2c(cn1)ncn2-c1ccc2c(cnn2C)c1"),
}
ANSWER_SMI = {
    "ADMET-001": ("fexofenadine", "CC(C)(C(=O)O)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1"),
    "ADMET-002": ("valacyclovir", "CC(C)[C@H](N)C(=O)OCCOCn1cnc2c(=O)nc(N)[nH]c21"),
    "ADMET-003": ("OXS008474",    "Cc1cc(F)ncc1-c1cc2c(cn1)ncn2-c1ccc2c(cnn2C)c1"),
}
LIABILITY = {
    "ADMET-001": "hERG",
    "ADMET-002": "oral_exposure",
    "ADMET-003": "solubility",
}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tokenize(smi: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    UNK = 1
    ids = [VOCAB.get(c, UNK) for c in smi[:max_len]]
    n = len(ids)
    ids = ids + [PAD] * (max_len - n)
    attn = np.zeros(max_len, dtype=bool)
    attn[:n] = True
    return np.asarray(ids, dtype=np.int64), attn


@torch.no_grad()
def predict_admet(model, smiles_list: list[str], device, max_len: int = 128, bs: int = 64):
    n_tasks = len(TDC_DATASETS)
    out = np.zeros((len(smiles_list), n_tasks), dtype=np.float32)
    is_cls = [t == "binary" for _, _, t in TDC_DATASETS]
    for i in range(0, len(smiles_list), bs):
        chunk = smiles_list[i:i+bs]
        ids_b, mask_b = [], []
        for smi in chunk:
            ids, mask = tokenize(smi, max_len)
            ids_b.append(ids); mask_b.append(mask)
        ids_t = torch.from_numpy(np.stack(ids_b)).to(device)
        mask_t = torch.from_numpy(np.stack(mask_b)).to(device)
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16 if device.type=="cuda" else torch.float32):
            logits = model(ids_t, mask_t).float().cpu().numpy()
        for t, cls in enumerate(is_cls):
            if cls:
                logits[:, t] = 1.0 / (1.0 + np.exp(-logits[:, t]))
        out[i:i+len(chunk)] = logits
    return out


def rdkit_descriptors(smiles_list: list[str]) -> pd.DataFrame:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, QED, AllChem, DataStructs
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
    # PAINS + Brenk filters (RDKit FilterCatalog)
    from rdkit.Chem import FilterCatalog
    pains_params = FilterCatalog.FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog.FilterCatalog(pains_params)
    brenk_params = FilterCatalog.FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    brenk_catalog = FilterCatalog.FilterCatalog(brenk_params)

    rows = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            rows.append({k: None for k in [
                "logP","TPSA","MW","HBD","HBA","RotBonds","AromRings","fsp3",
                "QED","pains_count","brenk_count","heavy_atoms"]})
            continue
        rows.append({
            "logP": Descriptors.MolLogP(m),
            "TPSA": Descriptors.TPSA(m),
            "MW":   Descriptors.MolWt(m),
            "HBD":  rdMolDescriptors.CalcNumHBD(m),
            "HBA":  rdMolDescriptors.CalcNumHBA(m),
            "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(m),
            "AromRings": rdMolDescriptors.CalcNumAromaticRings(m),
            "fsp3": rdMolDescriptors.CalcFractionCSP3(m),
            "QED":  QED.default(m),
            "pains_count":  len(pains_catalog.GetMatches(m)),
            "brenk_count":  len(brenk_catalog.GetMatches(m)),
            "heavy_atoms":  m.GetNumHeavyAtoms(),
        })
    return pd.DataFrame(rows)


def tanimoto_to(parent_smi: str, candidates: list[str]) -> list[float]:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    pm = Chem.MolFromSmiles(parent_smi)
    pfp = AllChem.GetMorganFingerprintAsBitVect(pm, 2, nBits=2048)
    out = []
    for smi in candidates:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            out.append(0.0); continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        out.append(DataStructs.TanimotoSimilarity(fp, pfp))
    return out


# Per-liability composite fitness (higher = better rescue candidate)
def composite_fitness(row: dict, liability: str) -> float:
    """row contains predicted ADMET (TDC task names lowercase mapping) + descriptors."""
    if liability == "hERG":
        # hERG rescue: low hERG probability, low logP, preserved Tanimoto in 0.4-0.8 band,
        # not introducing other liabilities (BBB, ClinTox, AMES).
        herg_p = row["pred_hERG"]
        logp = row["logP"] or 5.0
        clintox = row["pred_ClinTox"]
        ames = row["pred_AMES"]
        tan = row["tanimoto_to_parent"]
        tan_band = 1.0 - abs(0.6 - tan)  # peaks at 0.6
        return (
            -2.0 * herg_p
            - 0.5 * max(0, logp - 3.0)  # penalize logP above 3
            + 1.5 * tan_band
            - 1.0 * clintox
            - 0.8 * ames
            - 0.5 * (row["pains_count"] + row["brenk_count"])
        )
    if liability == "oral_exposure":
        # Oral exposure rescue: high HIA + Bioavailability + Caco2, low Pgp efflux,
        # preserved parent Tanimoto (prodrug shouldn't drift too far)
        hia = row["pred_HIA_Hou"]
        bioavail = row["pred_Bioavailability_Ma"]
        caco2 = row["pred_Caco2_Wang"]
        pgp = row["pred_Pgp_Broccatelli"]
        tan = row["tanimoto_to_parent"]
        tan_band = 1.0 - abs(0.5 - tan)
        return (
            +2.0 * hia
            + 1.5 * bioavail
            + 0.5 * caco2 / 10.0  # caco2 is regression value, scale
            - 1.0 * pgp
            + 1.0 * tan_band
            - 0.5 * (row["pains_count"] + row["brenk_count"])
        )
    if liability == "solubility":
        # Solubility rescue: high logS (Solubility_AqSolDB), low logP,
        # not increasing molecular weight too much, preserved Tanimoto
        logs = row["pred_Solubility_AqSolDB"]  # higher = more soluble
        logp_pred = row["pred_Lipophilicity_AstraZeneca"]
        logp_rdkit = row["logP"] or 5.0
        mw = row["MW"] or 0
        tan = row["tanimoto_to_parent"]
        tan_band = 1.0 - abs(0.6 - tan)
        return (
            +2.0 * (logs + 5.0) / 5.0   # logS typically -10 to 0; normalize
            - 0.6 * max(0, logp_rdkit - 3.0)
            - 0.5 * logp_pred / 5.0
            - 0.3 * max(0, mw - 500) / 100.0
            + 1.0 * tan_band
            - 0.5 * (row["pains_count"] + row["brenk_count"])
        )
    return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aux-ckpt", type=Path, required=True)
    p.add_argument("--cases", default="ADMET-001,ADMET-002,ADMET-003")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--results-base", type=Path, default=Path("rasyn/data/clean"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    # Load aux predictor
    _log(f"Loading aux predictor {args.aux_ckpt}")
    ckpt = torch.load(args.aux_ckpt, map_location=device, weights_only=False)
    cargs = ckpt.get("args", {})
    task_names = ckpt.get("task_names") or [n for _, n, _ in TDC_DATASETS]
    n_tasks = len(task_names)
    model = MultiTaskADMET(
        n_tasks=n_tasks,
        d_model=cargs.get("d_model", 768),
        n_heads=cargs.get("n_heads", 12),
        n_layers=cargs.get("n_layers", 8),
        max_len=cargs.get("max_len", 128),
    ).to(device)
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    _log(f"  aux model loaded ({n_tasks} tasks)")

    results_dirs = {
        "ADMET-001": args.results_base / "stage5_results_v3",
        "ADMET-002": args.results_base / "stage5_results_v3",
        "ADMET-003": args.results_base / "stage5_results_admet003",
    }

    summary = {}
    for case_id in args.cases.split(","):
        case_id = case_id.strip()
        liability = LIABILITY[case_id]
        parent_name, parent_smi = PARENT_SMI[case_id]
        ans_name, ans_smi = ANSWER_SMI[case_id]
        candidates_path = results_dirs[case_id] / f"{case_id}_top_candidates.parquet"
        if not candidates_path.exists():
            _log(f"  skipping {case_id}: {candidates_path} not found")
            continue
        _log(f"\n===== {case_id} ({liability}) =====")
        cands_df = pd.read_parquet(candidates_path).sort_values("rescue_score", ascending=False).reset_index(drop=True)
        topk = cands_df.head(args.top_k).copy()
        topk["source"] = "top_ranker"
        topk["rank_by_ranker"] = range(1, len(topk) + 1)

        # Append the literature answer
        ans_row = {
            "candidate_smiles": ans_smi,
            "rescue_score": None,
            "rescue_label": "literature_answer",
            "channel": "literature",
            "source": "literature_answer",
            "rank_by_ranker": None,
        }
        for c in topk.columns:
            if c not in ans_row:
                ans_row[c] = None
        eval_df = pd.concat([topk, pd.DataFrame([ans_row])], ignore_index=True)

        smiles = eval_df["candidate_smiles"].tolist()
        _log(f"  evaluating {len(smiles)} molecules (top {args.top_k} + 1 answer)")

        # Aux predictions
        admet = predict_admet(model, smiles, device)
        for t, name in enumerate(task_names):
            eval_df[f"pred_{name}"] = admet[:, t]

        # RDKit descriptors
        desc_df = rdkit_descriptors(smiles)
        for c in desc_df.columns:
            eval_df[c] = desc_df[c].values

        # Tanimoto to parent
        eval_df["tanimoto_to_parent"] = tanimoto_to(parent_smi, smiles)

        # Composite fitness
        rows_dicts = eval_df.to_dict(orient="records")
        eval_df["composite_fitness"] = [composite_fitness(r, liability) for r in rows_dicts]

        # Final ranking by composite fitness
        eval_df = eval_df.sort_values("composite_fitness", ascending=False).reset_index(drop=True)
        eval_df["rank_by_fitness"] = range(1, len(eval_df) + 1)

        # Find literature answer's rank by composite fitness
        lit_row = eval_df[eval_df["source"] == "literature_answer"].iloc[0]
        lit_rank = int(lit_row["rank_by_fitness"])
        lit_fitness = lit_row["composite_fitness"]
        best_row = eval_df.iloc[0]
        worst_top_row = eval_df.iloc[args.top_k]

        _log(f"  Literature answer ({ans_name}) ranks #{lit_rank} of {len(eval_df)} by composite fitness")
        _log(f"  Best composite fitness: {best_row['composite_fitness']:.3f} | answer: {lit_fitness:.3f}")
        _log(f"  Best candidate SMILES: {best_row['candidate_smiles'][:80]}")

        # Save full eval per case
        eval_df.to_parquet(args.out / f"{case_id}_evaluation.parquet", index=False)

        # Build summary
        summary[case_id] = {
            "liability": liability,
            "parent_name": parent_name,
            "literature_answer_name": ans_name,
            "n_evaluated": len(eval_df),
            "literature_answer_rank_by_fitness": lit_rank,
            "literature_answer_fitness": float(lit_fitness),
            "best_candidate_fitness": float(best_row["composite_fitness"]),
            "best_candidate_smiles": best_row["candidate_smiles"],
            "best_candidate_source": best_row["source"],
            "best_candidate_rank_by_ranker": (
                int(best_row["rank_by_ranker"]) if not pd.isna(best_row["rank_by_ranker"]) else None
            ),
            "answer_better_than_topk_pool": lit_rank <= args.top_k,
        }

        # Print top 5 + answer location
        print()
        print(f"  Top-5 by composite_fitness:")
        cols_to_show = ["rank_by_fitness", "rank_by_ranker", "source", "composite_fitness", "tanimoto_to_parent", "candidate_smiles"]
        for _, r in eval_df.head(5).iterrows():
            print(f"    #{int(r['rank_by_fitness'])} "
                  f"src={r['source']:<18} "
                  f"fit={r['composite_fitness']:+.3f} "
                  f"tan={r['tanimoto_to_parent']:.3f}  "
                  f"{r['candidate_smiles'][:70]}")
        print(f"  Literature answer at:")
        print(f"    #{lit_rank} src=literature_answer  "
              f"fit={lit_fitness:+.3f} "
              f"tan={lit_row['tanimoto_to_parent']:.3f}  "
              f"{ans_smi[:70]}")

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"\nDone. Output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
