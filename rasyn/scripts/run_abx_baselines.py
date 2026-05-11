"""All 10 ABX baselines per spec §18.1–§18.10.

Implements every baseline the spec requires and reports closed-mode rank
+ top-k for the hidden hit alongside the full ranker (Rasyn v1). The
output is a single comparison CSV + JSON summary that goes into the
sealed-case report alongside the system verdicts.

Run after Phase 7 (the v1 ranker already shipped + locked v1 predictions
in artifacts/abx_stage5_results/):
    python scripts/run_abx_baselines.py \\
        --library rasyn/data/clean/antibiotic/abx_molecules.parquet \\
        --facts   rasyn/data/clean/antibiotic/antibacterial_assay_facts.parquet \\
        --known   rasyn/data/clean/antibiotic/known_antibiotics.parquet  \\
        --rasyn-results artifacts/abx_stage5_results \\
        --cases ABX-001,ABX-002 \\
        --registry rasyn/rasyn/antibiotic/sealed_case_registry.yaml \\
        --out artifacts/abx_baselines

Note for D-MPNN (§18.7): we ship a small from-scratch message-passing
classifier rather than depending on the `chemprop` package, so the
baseline runs without extra installs. Architecture is intentionally
small (3 layers, hidden 64) so it does not become a stealth full model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- helpers ----------------------------------------------------------

def _morgan_fp(smi: str, nbits: int = 2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)


def _fp_array(smiles_list: list[str], nbits: int = 2048) -> tuple[np.ndarray, list[int]]:
    from rdkit.DataStructs import ConvertToNumpyArray
    out = np.zeros((len(smiles_list), nbits), dtype=np.uint8)
    valid: list[int] = []
    for i, smi in enumerate(smiles_list):
        fp = _morgan_fp(smi, nbits)
        if fp is None:
            continue
        arr = np.zeros(nbits, dtype=np.uint8)
        ConvertToNumpyArray(fp, arr)
        out[i] = arr
        valid.append(i)
    return out, valid


def _physchem_score(smi: str) -> float:
    """Drug-like proximity: 1 if all in Lipinski-ish ranges, else penalty."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return 0.0
        mw = Descriptors.MolWt(m)
        clogp = Descriptors.MolLogP(m)
        tpsa = Descriptors.TPSA(m)
        hba = Lipinski.NumHAcceptors(m)
        hbd = Lipinski.NumHDonors(m)
        score = 1.0
        for v, lo, hi in [(mw, 150, 700), (clogp, -2, 6), (tpsa, 10, 180), (hba, 0, 12), (hbd, 0, 6)]:
            if v < lo or v > hi:
                score -= 0.2
        return max(0.0, score)
    except Exception:
        return 0.0


def _tox_heuristic_score(smi: str) -> float:
    """Spec §18.5 — INTENTIONALLY DANGEROUS baseline: high score for reactive groups.
    If Rasyn does not beat this baseline on selectivity, the model is just
    learning to flag reactives.
    """
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return 0.0
        smarts_red_flags = ["[N+](=O)[O-]", "C=O.O=O", "[CH](Cl)Cl", "C=C-C=O",
                             "S(=O)(=O)O", "[Cr]", "[As]"]
        score = 0.0
        for sm in smarts_red_flags:
            patt = Chem.MolFromSmarts(sm)
            if patt is None:
                continue
            if m.HasSubstructMatch(patt):
                score += 0.2
        return min(1.0, score)
    except Exception:
        return 0.0


def _tanimoto_max(fp, ref_fps) -> float:
    from rdkit.DataStructs import TanimotoSimilarity
    if not ref_fps:
        return 0.0
    return max(TanimotoSimilarity(fp, rf) for rf in ref_fps)


# ---------- D-MPNN baseline --------------------------------------------------

def _train_dmpnn(actives_smiles: list[str], inactives_smiles: list[str]):
    """Tiny message-passing classifier — 3 layers, hidden 64. CPU-friendly.
    Returns a function: smi -> P(active)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rdkit import Chem

    DIM = 64
    PAD = 0

    def mol_to_tensors(smi: str):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        atoms = [a.GetAtomicNum() for a in m.GetAtoms()][:40]
        n = len(atoms)
        if n == 0:
            return None
        adj = np.zeros((40, 40), dtype=np.float32)
        for b in m.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i < 40 and j < 40:
                adj[i, j] = adj[j, i] = 1.0
        np.fill_diagonal(adj, 1.0)
        ax = np.zeros(40, dtype=np.int64)
        ax[:n] = atoms
        mask = np.zeros(40, dtype=np.float32); mask[:n] = 1.0
        return ax, adj, mask

    class DMPNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.atom_emb = nn.Embedding(120, DIM, padding_idx=PAD)
            self.gnn = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(3)])
            self.cls = nn.Linear(DIM, 1)
        def forward(self, atoms, adj, mask):
            x = self.atom_emb(atoms)  # (B, 40, DIM)
            adj_n = adj / (adj.sum(-1, keepdim=True).clamp(min=1.0))
            for layer in self.gnn:
                x = F.gelu(layer(torch.bmm(adj_n, x)))
            x = x * mask.unsqueeze(-1)
            x = x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
            return torch.sigmoid(self.cls(x).squeeze(-1))

    model = DMPNN()
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)

    def batchify(smis: list[str], labels: list[int]):
        atoms, adjs, masks, lbls = [], [], [], []
        for s, y in zip(smis, labels):
            t = mol_to_tensors(s)
            if t is None: continue
            ax, ad, mk = t
            atoms.append(ax); adjs.append(ad); masks.append(mk); lbls.append(float(y))
        if not atoms:
            return None
        return (torch.tensor(np.stack(atoms)), torch.tensor(np.stack(adjs)),
                torch.tensor(np.stack(masks)), torch.tensor(lbls))

    all_smis = actives_smiles + inactives_smiles
    all_lbls = [1] * len(actives_smiles) + [0] * len(inactives_smiles)
    perm = np.random.default_rng(0).permutation(len(all_smis))
    all_smis = [all_smis[i] for i in perm]
    all_lbls = [all_lbls[i] for i in perm]
    model.train()
    for ep in range(20):
        for i in range(0, len(all_smis), 64):
            batch = batchify(all_smis[i:i+64], all_lbls[i:i+64])
            if batch is None: continue
            ax, ad, mk, y = batch
            optim.zero_grad()
            p = model(ax, ad, mk)
            loss = F.binary_cross_entropy(p, y)
            loss.backward()
            optim.step()
    model.eval()

    def predict(smi: str) -> float:
        t = mol_to_tensors(smi)
        if t is None: return 0.0
        ax, ad, mk = t
        with torch.no_grad():
            p = model(torch.tensor(ax).unsqueeze(0), torch.tensor(ad).unsqueeze(0), torch.tensor(mk).unsqueeze(0))
        return float(p.item())
    return predict


# ---------- 10 baselines -----------------------------------------------------

@dataclass
class BaselineResult:
    case_id: str
    baseline: str
    library_size: int
    hidden_hit_rank: int | None
    top_1_pct: bool
    top_10: bool
    top_100: bool


def rank_random(cases, library_smiles, seed: int = 0) -> list[BaselineResult]:
    rng = np.random.default_rng(seed)
    out = []
    for case_id, ans_smi in cases:
        if ans_smi and ans_smi not in library_smiles:
            library_smiles = library_smiles + [ans_smi]
        perm = rng.permutation(len(library_smiles))
        rank = None
        if ans_smi:
            try:
                pos = library_smiles.index(ans_smi)
                rank = int(np.where(perm == pos)[0][0]) + 1
            except ValueError:
                rank = None
        out.append(_mk(case_id, "18.1_random", len(library_smiles), rank))
    return out


def rank_by_known_antibiotic_similarity(cases, library_smiles, known_smiles):
    fps, valid = _fp_array(library_smiles)
    known_fps = [_morgan_fp(s) for s in known_smiles if _morgan_fp(s) is not None]
    out = []
    for case_id, ans_smi in cases:
        if ans_smi:
            if ans_smi not in library_smiles:
                library_smiles = library_smiles + [ans_smi]
                fps, valid = _fp_array(library_smiles)
        scores = np.zeros(len(library_smiles))
        for i, smi in enumerate(library_smiles):
            fp = _morgan_fp(smi)
            scores[i] = _tanimoto_max(fp, known_fps) if fp else -1.0
        order = np.argsort(-scores)
        rank = _rank_of(library_smiles, order, ans_smi)
        out.append(_mk(case_id, "18.2_known_antibiotic_similarity", len(library_smiles), rank))
    return out


def rank_by_training_active_nn(cases, library_smiles, facts_df, organism_map):
    out = []
    for case_id, ans_smi in cases:
        organism = organism_map[case_id]
        actives = facts_df[(facts_df["organism"] == organism) & (facts_df["activity_label"] == "active")]
        a_smiles = actives["canonical_smiles"].dropna().astype(str).unique().tolist()[:200]
        if ans_smi and ans_smi in a_smiles:
            a_smiles.remove(ans_smi)  # treat answer as held out
        a_fps = [_morgan_fp(s) for s in a_smiles if _morgan_fp(s) is not None]
        lib = library_smiles[:]
        if ans_smi and ans_smi not in lib:
            lib.append(ans_smi)
        scores = np.array([_tanimoto_max(_morgan_fp(s), a_fps) if _morgan_fp(s) else -1.0 for s in lib])
        order = np.argsort(-scores)
        rank = _rank_of(lib, order, ans_smi)
        out.append(_mk(case_id, "18.3_training_active_nn", len(lib), rank))
    return out


def rank_by_physchem(cases, library_smiles):
    out = []
    for case_id, ans_smi in cases:
        lib = library_smiles[:]
        if ans_smi and ans_smi not in lib:
            lib.append(ans_smi)
        scores = np.array([_physchem_score(s) for s in lib])
        order = np.argsort(-scores)
        rank = _rank_of(lib, order, ans_smi)
        out.append(_mk(case_id, "18.4_physchem_heuristic", len(lib), rank))
    return out


def rank_by_tox_heuristic(cases, library_smiles):
    out = []
    for case_id, ans_smi in cases:
        lib = library_smiles[:]
        if ans_smi and ans_smi not in lib:
            lib.append(ans_smi)
        scores = np.array([_tox_heuristic_score(s) for s in lib])
        order = np.argsort(-scores)
        rank = _rank_of(lib, order, ans_smi)
        out.append(_mk(case_id, "18.5_tox_heuristic_DANGEROUS", len(lib), rank))
    return out


def rank_by_ecfp_classifier(cases, library_smiles, facts_df, organism_map, clf_kind: str):
    out = []
    for case_id, ans_smi in cases:
        organism = organism_map[case_id]
        actives = facts_df[(facts_df["organism"] == organism) & (facts_df["activity_label"] == "active")]
        inactives = facts_df[(facts_df["organism"] == organism) & (facts_df["activity_label"] == "inactive")]
        a_smi = actives["canonical_smiles"].dropna().astype(str).unique().tolist()
        i_smi = inactives["canonical_smiles"].dropna().astype(str).unique().tolist()
        if not a_smi or not i_smi:
            out.append(_mk(case_id, f"18.6_or_18.7_{clf_kind}", len(library_smiles), None))
            continue
        if ans_smi and ans_smi in a_smi:
            a_smi.remove(ans_smi)
        X_train = np.vstack([_fp_array(a_smi)[0], _fp_array(i_smi)[0]])
        y_train = np.concatenate([np.ones(len(a_smi)), np.zeros(len(i_smi))])
        from sklearn.ensemble import RandomForestClassifier
        try:
            from xgboost import XGBClassifier
        except ImportError:
            XGBClassifier = None
        if clf_kind == "rf":
            clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
            tag = "18.6_ecfp_rf"
        elif clf_kind == "xgb" and XGBClassifier is not None:
            clf = XGBClassifier(n_estimators=200, max_depth=6, random_state=0, n_jobs=-1, use_label_encoder=False, eval_metric="logloss")
            tag = "18.6_ecfp_xgb"
        else:
            clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
            tag = "18.6_ecfp_rf"
        clf.fit(X_train, y_train)
        lib = library_smiles[:]
        if ans_smi and ans_smi not in lib:
            lib.append(ans_smi)
        X_lib, _ = _fp_array(lib)
        scores = clf.predict_proba(X_lib)[:, 1]
        order = np.argsort(-scores)
        rank = _rank_of(lib, order, ans_smi)
        out.append(_mk(case_id, tag, len(lib), rank))
    return out


def rank_by_dmpnn(cases, library_smiles, facts_df, organism_map):
    out = []
    for case_id, ans_smi in cases:
        organism = organism_map[case_id]
        actives = facts_df[(facts_df["organism"] == organism) & (facts_df["activity_label"] == "active")]
        inactives = facts_df[(facts_df["organism"] == organism) & (facts_df["activity_label"] == "inactive")]
        a_smi = actives["canonical_smiles"].dropna().astype(str).unique().tolist()
        i_smi = inactives["canonical_smiles"].dropna().astype(str).unique().tolist()
        if not a_smi or not i_smi:
            out.append(_mk(case_id, "18.7_dmpnn", len(library_smiles), None))
            continue
        if ans_smi and ans_smi in a_smi:
            a_smi.remove(ans_smi)
        predict = _train_dmpnn(a_smi[:500], i_smi[:500])
        lib = library_smiles[:]
        if ans_smi and ans_smi not in lib:
            lib.append(ans_smi)
        scores = np.array([predict(s) for s in lib])
        order = np.argsort(-scores)
        rank = _rank_of(lib, order, ans_smi)
        out.append(_mk(case_id, "18.7_dmpnn", len(lib), rank))
    return out


def rank_by_organism_agnostic(cases, library_smiles, facts_df):
    """§18.8 — pool all active labels regardless of organism. Single global classifier."""
    actives = facts_df[facts_df["activity_label"] == "active"]
    inactives = facts_df[facts_df["activity_label"] == "inactive"]
    a_smi = actives["canonical_smiles"].dropna().astype(str).unique().tolist()
    i_smi = inactives["canonical_smiles"].dropna().astype(str).unique().tolist()
    out = []
    if not a_smi or not i_smi:
        for case_id, ans_smi in cases:
            out.append(_mk(case_id, "18.8_organism_agnostic", len(library_smiles), None))
        return out
    X_train = np.vstack([_fp_array(a_smi[:2000])[0], _fp_array(i_smi[:2000])[0]])
    y_train = np.concatenate([np.ones(min(2000, len(a_smi))), np.zeros(min(2000, len(i_smi)))])
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(X_train, y_train)
    for case_id, ans_smi in cases:
        lib = library_smiles[:]
        if ans_smi and ans_smi not in lib:
            lib.append(ans_smi)
        X_lib, _ = _fp_array(lib)
        scores = clf.predict_proba(X_lib)[:, 1]
        order = np.argsort(-scores)
        rank = _rank_of(lib, order, ans_smi)
        out.append(_mk(case_id, "18.8_organism_agnostic", len(lib), rank))
    return out


def rasyn_without_diffusion(rasyn_results_dir: Path) -> list[BaselineResult]:
    """§18.9 — Rasyn's v1 result IS this baseline (Ch-E/F empty in Phase 7)."""
    out = []
    for f in rasyn_results_dir.glob("*_closed_metrics.json"):
        data = json.loads(f.read_text())
        out.append(_mk(
            data["case_id"], "18.9_ranker_without_diffusion",
            data["library_size"], data["hidden_hit_rank"],
        ))
    return out


def rasyn_diffusion_without_selectivity_placeholder(cases, library_smiles) -> list[BaselineResult]:
    """§18.10 — needs trained diffusion ckpt to populate; reported as
    pending-training when no diffusion ckpt yet exists. We still emit a
    BaselineResult row so the comparison table is exhaustive."""
    out = []
    for case_id, ans_smi in cases:
        out.append(_mk(case_id, "18.10_diffusion_without_selectivity_pending_training", len(library_smiles), None))
    return out


# ---------- utility ----------------------------------------------------------

def _rank_of(lib_smiles: list[str], order: np.ndarray, target_smi: str | None) -> int | None:
    if not target_smi:
        return None
    try:
        target_idx = lib_smiles.index(target_smi)
    except ValueError:
        return None
    rank_pos = int(np.where(order == target_idx)[0][0]) + 1
    return rank_pos


def _mk(case_id: str, baseline: str, library_size: int, rank: int | None) -> BaselineResult:
    n = library_size
    top_1pct = rank is not None and rank <= max(1, n // 100)
    top_10 = rank is not None and rank <= 10
    top_100 = rank is not None and rank <= 100
    return BaselineResult(case_id, baseline, n, rank, top_1pct, top_10, top_100)


# ---------- main -------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--known", type=Path, default=None)
    p.add_argument("--registry", type=Path, default=Path("rasyn/rasyn/antibiotic/sealed_case_registry.yaml"))
    p.add_argument("--cases", default="ABX-001,ABX-002")
    p.add_argument("--rasyn-results", type=Path, default=None,
                   help="artifacts/abx_stage5_results dir for §18.9 (ranker without diffusion)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    _log("Loading library / facts / registry")
    lib_df = pd.read_parquet(args.library)
    library_smiles = lib_df["canonical_smiles"].dropna().astype(str).unique().tolist()
    facts_df = pd.read_parquet(args.facts)

    if args.known and args.known.exists():
        known_df = pd.read_parquet(args.known)
        known_smiles = known_df["canonical_smiles"].dropna().astype(str).unique().tolist()
    else:
        known_smiles = []

    reg = yaml.safe_load(args.registry.read_text())
    case_lookup = {c["case_id"]: c for c in reg["cases"]}
    cases = []
    organism_map = {}
    for cid in args.cases.split(","):
        cid = cid.strip()
        c = case_lookup.get(cid)
        if c is None:
            continue
        ans = (c.get("hidden_solution") or {}).get("canonical_smiles")
        cases.append((cid, ans))
        organism_map[cid] = (c.get("organism_context") or {}).get("organism", "unknown")

    _log(f"cases={[c[0] for c in cases]} library_size={len(library_smiles)}")

    all_results: list[BaselineResult] = []

    _log("§18.1 random ranking")
    all_results += rank_random(cases, library_smiles)

    _log("§18.2 known-antibiotic similarity")
    all_results += rank_by_known_antibiotic_similarity(cases, library_smiles, known_smiles)

    _log("§18.3 NN to training actives")
    all_results += rank_by_training_active_nn(cases, library_smiles, facts_df, organism_map)

    _log("§18.4 physicochemical heuristic")
    all_results += rank_by_physchem(cases, library_smiles)

    _log("§18.5 tox/reactivity heuristic (DANGEROUS)")
    all_results += rank_by_tox_heuristic(cases, library_smiles)

    _log("§18.6 ECFP + RF / XGB")
    all_results += rank_by_ecfp_classifier(cases, library_smiles, facts_df, organism_map, "rf")
    all_results += rank_by_ecfp_classifier(cases, library_smiles, facts_df, organism_map, "xgb")

    _log("§18.7 D-MPNN (tiny from-scratch)")
    all_results += rank_by_dmpnn(cases, library_smiles, facts_df, organism_map)

    _log("§18.8 organism-agnostic classifier")
    all_results += rank_by_organism_agnostic(cases, library_smiles, facts_df)

    if args.rasyn_results and args.rasyn_results.exists():
        _log("§18.9 ranker without diffusion (Rasyn v1 result)")
        all_results += rasyn_without_diffusion(args.rasyn_results)

    _log("§18.10 diffusion without selectivity (pending training)")
    all_results += rasyn_diffusion_without_selectivity_placeholder(cases, library_smiles)

    df = pd.DataFrame([asdict(r) for r in all_results])
    df.to_csv(args.out / "baselines_comparison.csv", index=False)
    (args.out / "baselines_comparison.json").write_text(df.to_json(orient="records", indent=2))
    _log(f"Wrote {args.out}/baselines_comparison.csv ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
