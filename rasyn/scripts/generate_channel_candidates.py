"""Generate Channel 4 or Channel 5 candidates on the pod that has the ckpt.

Runs locally on Pod C (for ch4) or Pod D (for ch5), outputs a tiny JSON
that can be SCP'd to Pod A for the Stage-5 scoring run. Avoids transferring
the 1.3 GB checkpoint cross-pod.

Run on Pod C:
    cd ~/wolverine/rasyn && source .venv/bin/activate
    python scripts/generate_channel_candidates.py \\
        --ckpt rasyn/data/clean/channel4_inverse_delta/checkpoint.pt \\
        --channel-name learned_inverse_delta \\
        --registry rasyn/data/registry/sealed_case_registry.yaml \\
        --cases ADMET-001,ADMET-002 \\
        --n-samples 200 \\
        --out /tmp/ch4_candidates.json

Run on Pod D:
    same but --ckpt channel5_forward_reward/checkpoint.pt --channel-name forward_reward_generator
        --out /tmp/ch5_candidates.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import torch


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--channel-name", required=True)
    p.add_argument("--registry", type=Path,
                   default=Path("rasyn/data/registry/sealed_case_registry.yaml"))
    p.add_argument("--cases", default="ADMET-001,ADMET-002")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    # Reuse the channel4_5_sample function defined in stage5_inference
    # (it's a free-standing function that loads ckpt + samples)
    from stage5_inference import channel4_5_sample

    reg = yaml.safe_load(args.registry.read_text())
    cases_by_id = {c["case_id"]: c for c in reg["cases"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    results: dict = {"channel": args.channel_name, "ckpt": str(args.ckpt), "cases": {}}
    for case_id in args.cases.split(","):
        case_id = case_id.strip()
        case = cases_by_id.get(case_id)
        if not case:
            _log(f"  case {case_id} not in registry; skipping")
            continue
        parent = case.get("parent") or case.get("parent_compound", {})
        parent_smiles = parent.get("canonical_smiles")
        liability_type = case.get("liability_type") or "unknown"
        if not parent_smiles:
            _log(f"  {case_id} parent SMILES null; skipping")
            continue
        _log(f"[{case_id}] parent={parent_smiles[:60]}... liability={liability_type}")
        samples = channel4_5_sample(
            parent_smiles, liability_type,
            ckpt_path=args.ckpt, device=device,
            n_samples=args.n_samples,
            temperature=args.temperature,
            channel_name=args.channel_name,
        )
        # We only need (candidate_smiles, channel) for the merge later
        results["cases"][case_id] = {
            "parent_smiles": parent_smiles,
            "liability_type": liability_type,
            "n_samples_requested": args.n_samples,
            "n_valid": len(samples),
            "candidates": [s["candidate_smiles"] for s in samples],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    _log(f"Saved {args.out} | cases={list(results['cases'].keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
