"""Train the conditions predictor (RETRO_PLAN R-3 Lock 7).

4-head classifier:
  - solvent_class    (30 classes)
  - catalyst_class   (20 classes)
  - temperature_bin  (4 bins: rt / reflux / cryo / high-T)
  - reagent_class    (multi-label over 50 buckets)

Shared 200M-MLM encoder of the (reactant_smiles . product_smiles) concat;
FiLM gain+bias from reaction-class one-hot.

Source data is from R-1 outputs: each reaction row carries the parsed
solvent_class / catalyst_class / temperature_bin / reagent_classes fields
(see rasyn/rasyn/synth/retro/schemas.py).

Run on 2x A100 (~8-16 GPU-h):
    torchrun --nproc_per_node=2 --standalone scripts/train_retro_conditions.py \\
        --pretrain checkpoints/smiles_lm_200m/checkpoint.pt \\
        --reactions rasyn/data/clean/retro/reactions_silver.parquet \\
        --steps 60000 --bs 64 --lr 2e-4 \\
        --out checkpoints/retro_conditions_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("train_retro_conditions")


SOLVENT_BUCKETS = [
    "DMSO", "DMF", "DMAc", "NMP", "THF", "2-MeTHF", "dioxane", "MeCN", "DCM", "DCE",
    "EtOH", "MeOH", "iPrOH", "water", "toluene", "xylene", "ether", "EtOAc",
    "acetone", "HFIP", "TFA_neat", "pyridine", "hexane", "heptane",
    "ionic_liquid", "supercritical_CO2", "neat", "solvent_free", "unknown", "other",
]
CATALYST_BUCKETS = [
    "Pd_phosphine", "Pd_NHC", "Ni_phosphine", "Cu", "Ru", "Rh", "Ir", "Pt",
    "organocat_amine", "organocat_acid", "Lewis_acid", "Bronsted_acid",
    "Bronsted_base", "phase_transfer", "photocat", "enzymatic",
    "none", "unknown", "other",
]
TEMPERATURE_BUCKETS = ["cryo", "rt", "warm", "reflux", "high_T", "unknown"]
REAGENT_BUCKETS = [
    "carbodiimide", "HATU_HBTU_family", "T3P", "boronic_acid", "boronate_ester",
    "aryl_halide", "alkyl_halide", "amine_primary", "amine_secondary",
    "carbonyl_protect", "amine_protect", "alcohol_protect",
    "reductant_NaBH4_class", "reductant_LiAlH4_class", "reductant_H2",
    "oxidant_DMP_class", "oxidant_Swern_class",
    "base_NaH", "base_K2CO3_class", "base_TEA_DIPEA", "base_DBU",
    "acid_HCl", "acid_TFA", "acid_sulfonic",
    "azide_source", "alkyne", "halogenation_NBS_class",
    "no_extra_reagent", "unknown", "other",
]
CLASS_BUCKETS = [
    "amide_coupling", "suzuki_coupling", "buchwald_hartwig", "reductive_amination",
    "sn2", "sn_ar", "negishi", "wittig", "click",
    "protection_deprotection", "other_cross_coupling", "unclassified",
]

SOL_TO_IDX = {s: i for i, s in enumerate(SOLVENT_BUCKETS)}
CAT_TO_IDX = {s: i for i, s in enumerate(CATALYST_BUCKETS)}
TEMP_TO_IDX = {s: i for i, s in enumerate(TEMPERATURE_BUCKETS)}
REA_TO_IDX = {s: i for i, s in enumerate(REAGENT_BUCKETS)}
CLASS_TO_IDX = {s: i for i, s in enumerate(CLASS_BUCKETS)}


def _maybe_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", type=Path, required=True)
    p.add_argument("--reactions", nargs="+", type=Path, required=True)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        torch, nn, F = _maybe_torch()
    except ImportError:
        logger.error("torch not installed; cannot train. Run on GPU pod.")
        return 1

    SCRIPTS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPTS_DIR))
    from h200_smiles_lm_pretrain import VOCAB, VOCAB_SIZE, PAD  # type: ignore

    def tokenize(reactant_str, product_str, max_len):
        s = reactant_str + ">>" + product_str
        ids = [VOCAB.get(c, 1) for c in s[:max_len]]
        n = len(ids)
        ids = ids + [PAD] * (max_len - n)
        mask = [True] * n + [False] * (max_len - n)
        return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=bool)

    import pyarrow.parquet as pq
    reactions = []
    for path in args.reactions:
        if path.exists():
            reactions.extend(pq.read_table(path).to_pylist())
    logger.info("loaded %d reactions", len(reactions))

    examples = []
    for r in reactions:
        prod = r.get("product_smiles") or r.get("product")
        reactants = r.get("reactant_smiles") or r.get("reactants") or []
        if not prod or not reactants:
            continue
        sol = r.get("solvent_class") or "unknown"
        cat = r.get("catalyst_class") or "unknown"
        tb = r.get("temperature_bin") or "unknown"
        rcs = r.get("reagent_classes") or []
        rc = r.get("reaction_class") or "unclassified"
        examples.append({
            "src": ".".join(reactants),
            "prod": prod,
            "sol": SOL_TO_IDX.get(sol, SOL_TO_IDX["unknown"]),
            "cat": CAT_TO_IDX.get(cat, CAT_TO_IDX["unknown"]),
            "temp": TEMP_TO_IDX.get(tb, TEMP_TO_IDX["unknown"]),
            "reagent_multilabel": np.array([1.0 if name in rcs else 0.0
                                              for name in REAGENT_BUCKETS], dtype=np.float32),
            "rc": CLASS_TO_IDX.get(rc, CLASS_TO_IDX["unclassified"]),
        })
    logger.info("built %d examples", len(examples))
    if not examples:
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain = torch.load(args.pretrain, map_location="cpu", weights_only=False)
    cargs = pretrain.get("args", {})
    d_model = cargs.get("d_model", 768)
    n_heads = cargs.get("n_heads", 12)
    n_layers = cargs.get("n_layers", 8)

    class FiLM(nn.Module):
        def __init__(self, d, k):
            super().__init__()
            self.g = nn.Embedding(k, d); self.b = nn.Embedding(k, d)
            nn.init.ones_(self.g.weight); nn.init.zeros_(self.b.weight)
        def forward(self, h, cidx):
            return self.g(cidx).unsqueeze(1) * h + self.b(cidx).unsqueeze(1)

    class ConditionsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
            self.pos_emb = nn.Embedding(args.max_len, d_model)
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, 0.1,
                                                 batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.film = FiLM(d_model, len(CLASS_BUCKETS))
            self.head_sol = nn.Linear(d_model, len(SOLVENT_BUCKETS))
            self.head_cat = nn.Linear(d_model, len(CATALYST_BUCKETS))
            self.head_temp = nn.Linear(d_model, len(TEMPERATURE_BUCKETS))
            self.head_rea = nn.Linear(d_model, len(REAGENT_BUCKETS))

        def forward(self, ids, mask, cidx):
            pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
            h = self.tok_emb(ids) + self.pos_emb(pos)
            h = self.encoder(h, src_key_padding_mask=~mask)
            h = self.film(h, cidx)
            h = self.norm(h)
            pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            return (self.head_sol(pooled), self.head_cat(pooled),
                    self.head_temp(pooled), self.head_rea(pooled))

    model = ConditionsModel().to(device)
    own = model.state_dict()
    for k, v in (pretrain.get("model") or {}).items():
        kk = k.removeprefix("module.")
        if kk in own and own[kk].shape == v.shape:
            own[kk] = v
    model.load_state_dict(own, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def get_batch():
        idxs = np.random.choice(len(examples), args.bs, replace=False)
        ids_list, mask_list = [], []
        sol_list, cat_list, temp_list, rea_list, cidx_list = [], [], [], [], []
        for i in idxs:
            ex = examples[i]
            ids, mask = tokenize(ex["src"], ex["prod"], args.max_len)
            ids_list.append(ids); mask_list.append(mask)
            sol_list.append(ex["sol"]); cat_list.append(ex["cat"])
            temp_list.append(ex["temp"]); rea_list.append(ex["reagent_multilabel"])
            cidx_list.append(ex["rc"])
        return (
            torch.from_numpy(np.stack(ids_list)).to(device),
            torch.from_numpy(np.stack(mask_list)).to(device),
            torch.tensor(sol_list, dtype=torch.long, device=device),
            torch.tensor(cat_list, dtype=torch.long, device=device),
            torch.tensor(temp_list, dtype=torch.long, device=device),
            torch.from_numpy(np.stack(rea_list)).to(device),
            torch.tensor(cidx_list, dtype=torch.long, device=device),
        )

    model.train()
    log_path = args.out / "training_log.jsonl"
    log_path.touch()
    t0 = time.time()
    for step in range(args.steps):
        ids, mask, sol_lbl, cat_lbl, temp_lbl, rea_lbl, cidx = get_batch()
        l_sol, l_cat, l_temp, l_rea = model(ids, mask, cidx)
        loss_sol = F.cross_entropy(l_sol, sol_lbl)
        loss_cat = F.cross_entropy(l_cat, cat_lbl)
        loss_temp = F.cross_entropy(l_temp, temp_lbl)
        loss_rea = F.binary_cross_entropy_with_logits(l_rea, rea_lbl)
        loss = loss_sol + loss_cat + loss_temp + 0.5 * loss_rea
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"step": step, "loss": float(loss.item()),
                                      "loss_sol": float(loss_sol.item()),
                                      "loss_cat": float(loss_cat.item()),
                                      "loss_temp": float(loss_temp.item()),
                                      "loss_rea": float(loss_rea.item()),
                                      "t": time.time() - t0}) + "\n")
            logger.info("step %d loss=%.4f", step, loss.item())

    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "buckets": {
            "solvent": SOLVENT_BUCKETS, "catalyst": CATALYST_BUCKETS,
            "temperature": TEMPERATURE_BUCKETS, "reagent": REAGENT_BUCKETS,
            "reaction_class": CLASS_BUCKETS,
        },
    }, args.out / "checkpoint.pt")
    logger.info("wrote %s", args.out / "checkpoint.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
