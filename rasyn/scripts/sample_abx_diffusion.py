"""Sample candidate SMILES from a trained ABX diffusion checkpoint.

Loads a diffusion ckpt produced by `train_abx_diffusion.py`, samples N
candidates per (organism, spectrum, selectivity) conditioning combo at a
range of guidance scales (per spec §11.6), validates with RDKit, and
writes a candidates JSON for `run_abx_sealed_cases.py --ch-e-json`.

Used by Channel E (fragment-conditioned / unconditional sampling) and
Channel F (selectivity-aware sampling). Channel E and Channel F differ
only in the conditioning vector + guidance scale: Channel F flips
`selectivity_label=selective` and uses guidance_scale > 1.0 to steer
generation toward selective antibacterials.

Run on a single GPU pod (~10 min for 300 samples × 3 cases):
    python scripts/sample_abx_diffusion.py \\
        --ckpt rasyn/data/clean/abx_diffusion/stage_5/checkpoint.pt \\
        --channel-name E_fragment_diffusion \\
        --registry rasyn/rasyn/antibiotic/sealed_case_registry.yaml \\
        --cases ABX-001,ABX-002,ABX-003 \\
        --n-samples 300 --guidance 1.0 --selectivity selective \\
        --out /tmp/abx_ch_e_candidates.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rasyn.antibiotic.graph_diffusion import (  # noqa: E402
    COND_DIM,
    AbsorbingDiffusion,
    GraphDenoiser,
    build_condition_vector,
    sample_graphs,
)
from rasyn.antibiotic.graph_io import MAX_ATOMS, graph_to_smiles  # noqa: E402


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_denoiser(ckpt_path: Path, device: torch.device) -> tuple[GraphDenoiser, AbsorbingDiffusion]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    d = GraphDenoiser(
        d_node=ckpt.get("d_node", 256), d_edge=ckpt.get("d_edge", 64),
        n_heads=ckpt.get("n_heads", 8), n_layers=ckpt.get("n_layers", 6),
        T=ckpt.get("T", 500), cond_dim=COND_DIM,
    ).to(device).eval()
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    d.load_state_dict(sd, strict=False)
    diff = AbsorbingDiffusion(T=ckpt.get("T", 500), device=device)
    return d, diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--channel-name", required=True)
    p.add_argument("--registry", type=Path,
                   default=Path("rasyn/rasyn/antibiotic/sealed_case_registry.yaml"))
    p.add_argument("--cases", default="ABX-001,ABX-002,ABX-003")
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--antibacterial", default="active")
    p.add_argument("--selectivity", default="selective")  # use 'unknown' for Channel E unconditional
    p.add_argument("--mean-n-atoms", type=int, default=24)
    p.add_argument("--std-n-atoms", type=int, default=6)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}; loading {args.ckpt}")
    denoiser, diffusion = load_denoiser(args.ckpt, device)

    reg = yaml.safe_load(args.registry.read_text())
    cases_by_id = {c["case_id"]: c for c in reg["cases"]}

    results = {"channel": args.channel_name, "ckpt": str(args.ckpt),
                "guidance_scale": args.guidance, "cases": {}}
    for case_id in args.cases.split(","):
        case_id = case_id.strip()
        case = cases_by_id.get(case_id)
        if not case:
            _log(f"  {case_id} missing in registry; skipping")
            continue
        org_ctx = case.get("organism_context") or {}
        organism = org_ctx.get("organism", "unknown")
        spectrum = org_ctx.get("spectrum_goal", "unknown")
        cond = build_condition_vector(
            organism=organism, spectrum=spectrum,
            antibacterial=args.antibacterial, selectivity=args.selectivity,
        ).to(device)
        _log(f"[{case_id}] organism={organism} spectrum={spectrum} guidance={args.guidance}")

        valid_smiles: list[str] = []
        n_done = 0
        torch.manual_seed(hash(case_id) & 0xFFFF)
        while n_done < args.n_samples:
            B = min(args.batch, args.n_samples - n_done)
            n_atoms = (torch.randn(B) * args.std_n_atoms + args.mean_n_atoms).clamp(8, MAX_ATOMS).long()
            cond_b = cond.unsqueeze(0).expand(B, -1).contiguous()
            node, edge, mask = sample_graphs(
                denoiser, diffusion, cond=cond_b,
                n_atoms_per_sample=n_atoms,
                guidance_scale=args.guidance, device=device,
            )
            node = node.cpu().numpy()
            edge = edge.cpu().numpy()
            mask = mask.cpu().numpy()
            for i in range(B):
                s = graph_to_smiles(node[i], edge[i], mask[i])
                if s and "." not in s and len(s) <= 200:  # no fragments, sane length
                    valid_smiles.append(s)
            n_done += B
        # Dedupe
        valid_smiles = list(dict.fromkeys(valid_smiles))
        results["cases"][case_id] = {
            "organism": organism, "spectrum": spectrum,
            "guidance_scale": args.guidance,
            "n_samples_requested": args.n_samples,
            "n_valid_returned": len(valid_smiles),
            "candidates": valid_smiles,
        }
        _log(f"  -> {len(valid_smiles)} valid unique candidates")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    _log(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
