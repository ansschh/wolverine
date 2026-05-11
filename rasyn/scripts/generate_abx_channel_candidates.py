"""Generate ABX Channel E or F candidates on the pod that has the ckpt.

Runs on Pod B (Ch-E) or Pod D (Ch-F) — same pattern as the ADMET
`generate_channel_candidates.py`. Output is a small JSON SCP'd to Pod A
for the sealed-case inference run; avoids transferring the ~770 MB
checkpoint cross-pod.

Run on Pod B (Ch-E):
    cd ~/wolverine/rasyn && source .venv/bin/activate
    python scripts/generate_abx_channel_candidates.py \\
        --ckpt rasyn/data/clean/abx_channel_e/checkpoint.pt \\
        --channel-name E_fragment_diffusion \\
        --registry rasyn/rasyn/antibiotic/sealed_case_registry.yaml \\
        --cases ABX-001,ABX-002,ABX-003 \\
        --n-samples 300 \\
        --out /tmp/abx_ch_e_candidates.json

Run on Pod D (Ch-F):
    same but --ckpt abx_channel_f/checkpoint.pt --channel-name F_edit_diffusion
        --out /tmp/abx_ch_f_candidates.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from train_abx_channel_e import (  # noqa: E402
    ABXChannelESeq2Seq,
    EXTENDED_VOCAB_SIZE,
    ORG_TOKEN_ID,
    ORGANISM_TYPE2STR,
    BOS,
    EOS,
    SEP,
    PAD,
    encode_input,
)
from h200_smiles_lm_pretrain import VOCAB  # noqa: E402

INV_VOCAB = {v: k for k, v in VOCAB.items()}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def detok(ids: list[int]) -> str:
    chars = []
    for t in ids:
        if t == EOS or t == PAD:
            break
        if t == BOS or t == SEP or t >= EXTENDED_VOCAB_SIZE:
            continue
        if t in INV_VOCAB:
            chars.append(INV_VOCAB[t])
    return "".join(chars)


@torch.no_grad()
def sample_for_organism(
    model: ABXChannelESeq2Seq,
    organism: str,
    *,
    n_samples: int,
    temperature: float,
    device: torch.device,
    dec_max_len: int = 130,
    enc_max_len: int = 16,
) -> list[str]:
    enc_ids, enc_mask = encode_input(organism, enc_max_len)
    enc_ids_t = torch.tensor([enc_ids], dtype=torch.long, device=device).expand(n_samples, -1).contiguous()
    enc_mask_t = torch.tensor([enc_mask], dtype=torch.bool, device=device).expand(n_samples, -1).contiguous()

    # Init decoder with BOS
    out_ids = torch.full((n_samples, dec_max_len), PAD, dtype=torch.long, device=device)
    out_ids[:, 0] = BOS
    out_mask = torch.zeros((n_samples, dec_max_len), dtype=torch.bool, device=device)
    out_mask[:, 0] = True

    # encode once
    T = enc_ids_t.size(1)
    pos = torch.arange(T, device=device).unsqueeze(0).expand_as(enc_ids_t)
    x = model.tok_emb(enc_ids_t) + model.pos_emb(pos)
    memory = model.encoder(x, src_key_padding_mask=~enc_mask_t)
    memory = model.norm(memory)

    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)

    for t in range(1, dec_max_len):
        dpos = torch.arange(t, device=device).unsqueeze(0).expand(n_samples, -1)
        y = model.tok_emb(out_ids[:, :t]) + model.pos_emb(dpos)
        causal = torch.triu(torch.full((t, t), float("-inf"), device=device), diagonal=1)
        y = model.decoder(
            y, memory, tgt_mask=causal, memory_key_padding_mask=~enc_mask_t,
            tgt_key_padding_mask=~out_mask[:, :t],
        )
        y = model.dec_norm(y)
        logits = model.lm_head(y)[:, -1] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
        nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
        out_ids[:, t] = nxt
        out_mask[:, t] = nxt != PAD
        finished = finished | (nxt == EOS)
        if finished.all():
            break

    smiles = []
    for i in range(n_samples):
        s = detok(out_ids[i, 1:].tolist())
        if s:
            smiles.append(s)
    return smiles


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--channel-name", required=True)
    p.add_argument("--registry", type=Path,
                   default=Path("rasyn/rasyn/antibiotic/sealed_case_registry.yaml"))
    p.add_argument("--cases", default="ABX-001,ABX-002,ABX-003")
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    reg = yaml.safe_load(args.registry.read_text())
    cases_by_id = {c["case_id"]: c for c in reg["cases"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    _log(f"Loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cargs = ckpt.get("args", {})
    d_model = cargs.get("d_model", 1024) if isinstance(cargs, dict) else 1024
    n_heads = cargs.get("n_heads", 16) if isinstance(cargs, dict) else 16
    n_layers = cargs.get("n_layers", 16) if isinstance(cargs, dict) else 16
    dec_len = cargs.get("dec_len", 130) if isinstance(cargs, dict) else 130
    enc_len = cargs.get("enc_len", 16) if isinstance(cargs, dict) else 16

    model = ABXChannelESeq2Seq(EXTENDED_VOCAB_SIZE, d_model, n_heads, n_layers,
                                max_len=max(dec_len, enc_len)).to(device)
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    results: dict = {"channel": args.channel_name, "ckpt": str(args.ckpt), "cases": {}}
    for case_id in args.cases.split(","):
        case_id = case_id.strip()
        case = cases_by_id.get(case_id)
        if not case:
            _log(f"  case {case_id} not in registry; skipping")
            continue
        org_ctx = case.get("organism_context") or {}
        organism = org_ctx.get("organism", "unknown")
        if organism not in ORGANISM_TYPE2STR:
            _log(f"  {case_id} organism={organism} unknown; using [ORG_UNKNOWN]")
            organism = "unknown"
        _log(f"[{case_id}] organism={organism} n_samples={args.n_samples}")
        samples = sample_for_organism(
            model, organism,
            n_samples=args.n_samples, temperature=args.temperature,
            device=device, dec_max_len=dec_len, enc_max_len=enc_len,
        )
        results["cases"][case_id] = {
            "organism": organism,
            "n_samples_requested": args.n_samples,
            "n_returned": len(samples),
            "candidates": samples,
        }
        _log(f"  -> {len(samples)} candidates")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    _log(f"Saved {args.out} | cases={list(results['cases'].keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
