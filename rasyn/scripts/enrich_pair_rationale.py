"""P-1e enrichment: ChEMBL silver-pair rationale via vLLM Llama-3.3-70B.

For each row in rescue_pair_candidates.parquet (silver tier), call vLLM with
structure + metric inputs (NO paper text), get back PairRationale JSON,
append to chembl_pair_rationale.parquet.

This is the immediate productive use of vLLM — it works on data we already
have, no paper-fetching dependency.

Run on the vLLM pod (avoids tunnel + slow upload):
    cd ~/wolverine/rasyn
    python scripts/enrich_pair_rationale.py \\
        --input rasyn/data/clean/rescue_pair_candidates.parquet \\
        --tier silver \\
        --vllm-url http://localhost:8000 \\
        --bs 8 \\
        --max-pairs 10000

Or from local with SSH tunnel:
    ssh -L 8000:localhost:8000 root@154.54.102.45 -p 18129 -i ~/.ssh/id_ed25519 -N
    python scripts/enrich_pair_rationale.py --vllm-url http://localhost:8000 ...

Output:
    chembl_pair_rationale.parquet
        Cols: pair_id, transformation_class (list), liability_driver (list),
              preserved_activity_features (list), expected_mechanism_improvement,
              expected_mechanism_retention, evidence_strength, warnings (list),
              extraction_runtime_ms, prompt_sha256

Concurrent vLLM requests via thread pool. AWQ Llama-3.3-70B on 1xA100 80GB
serves ~1-3 req/sec at this prompt size; expect ~1-3 hr for 10K pairs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROMPT_PATH = Path("rasyn/papers/rationale_prompt.md")
PROMPT_LOCK = Path("rasyn/papers/rationale_prompt.lock.json")

DEFAULT_VLLM = "http://localhost:8000"
DEFAULT_MODEL = "casperhansen/llama-3.3-70b-instruct-awq"


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_prompt(md: str) -> tuple[str, str]:
    sys_m = re.search(r"##\s+System message\s*\n+```\n(.*?)\n```", md, re.DOTALL)
    usr_m = re.search(
        r"##\s+Per-pair user message \(template\)\s*\n.*?```\n(.*?)\n```", md, re.DOTALL
    )
    if not sys_m or not usr_m:
        raise SystemExit("Failed to parse rationale_prompt.md system/user blocks.")
    return sys_m.group(1), usr_m.group(1)


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def render_user(template: str, row: dict) -> str:
    """Substitute {{KEY}} placeholders from a row dict."""
    out = template
    for key, val in row.items():
        out = out.replace(f"{{{{{key}}}}}", str(val))
    return out


def build_row_for_prompt(pair: dict) -> dict:
    """Map parquet pair row to the prompt template variable names."""
    def _fmt_num(x, ndp: int = 3) -> str:
        if x is None or pd.isna(x):
            return "unknown"
        try:
            return f"{float(x):.{ndp}f}"
        except Exception:
            return "unknown"

    return {
        "PAIR_ID": pair.get("pair_id", "unknown"),
        "PARENT_SMILES": pair.get("parent_smiles", ""),
        "CANDIDATE_SMILES": pair.get("candidate_smiles", ""),
        "LIABILITY_TYPE": pair.get("liability_type") or "unknown",
        "LIABILITY_ENDPOINT": pair.get("liability_endpoint") or "unknown",
        "TARGET": pair.get("target_chembl_id") or "unknown",
        "PARENT_ACTIVITY_PCHEMBL": _fmt_num(pair.get("parent_activity_pchembl")),
        "CANDIDATE_ACTIVITY_PCHEMBL": _fmt_num(pair.get("candidate_activity_pchembl")),
        "PARENT_LIABILITY_VALUE": _fmt_num(pair.get("parent_liability_value")),
        "CANDIDATE_LIABILITY_VALUE": _fmt_num(pair.get("candidate_liability_value")),
        "ACTIVITY_RETENTION": pair.get("activity_retention_bucket") or "unknown",
        "LIABILITY_IMPROVEMENT": pair.get("liability_improvement_category") or "unknown",
        "MURCKO_MATCH": str(bool(pair.get("murcko_match", False))).lower(),
        "HEAVY_ATOM_DIFF": str(pair.get("heavy_atom_diff") or "unknown"),
        "ECFP_TANIMOTO": _fmt_num(pair.get("ecfp_tanimoto"), ndp=3),
    }


def call_vllm(
    *,
    vllm_url: str,
    model_id: str,
    system_msg: str,
    user_msg: str,
    schema: dict,
    timeout: int = 120,
) -> tuple[dict, int]:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "PairRationale", "schema": schema, "strict": True},
        },
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    url = vllm_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    runtime_ms = int((time.time() - t0) * 1000)
    return data, runtime_ms


def process_one_pair(
    pair: dict,
    *,
    system_msg: str,
    user_template: str,
    schema: dict,
    vllm_url: str,
    model_id: str,
    prompt_sha: str,
) -> dict:
    """Process one pair through vLLM. Returns flat row for parquet."""
    row_vars = build_row_for_prompt(pair)
    user_msg = render_user(user_template, row_vars)
    try:
        resp, runtime_ms = call_vllm(
            vllm_url=vllm_url, model_id=model_id, system_msg=system_msg,
            user_msg=user_msg, schema=schema,
        )
        content = resp["choices"][0]["message"]["content"]
        rationale = json.loads(content)
        return {
            "pair_id": pair.get("pair_id"),
            "transformation_class": rationale.get("transformation_class", []),
            "liability_driver": rationale.get("liability_driver", []),
            "preserved_activity_features": rationale.get("preserved_activity_features", []),
            "expected_mechanism_improvement": rationale.get("expected_mechanism", {}).get(
                "liability_improvement", ""
            ),
            "expected_mechanism_retention": rationale.get("expected_mechanism", {}).get(
                "activity_retention", ""
            ),
            "evidence_strength": rationale.get("evidence_strength", "uncertain"),
            "warnings": rationale.get("warnings", []),
            "extraction_runtime_ms": runtime_ms,
            "prompt_sha256": prompt_sha,
            "model_id": model_id,
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": None,
        }
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        return {
            "pair_id": pair.get("pair_id"),
            "error": f"http: {e}",
            "extraction_runtime_ms": None,
        }
    except Exception as e:
        return {
            "pair_id": pair.get("pair_id"),
            "error": f"parse: {e}",
            "extraction_runtime_ms": None,
        }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="rescue_pair_candidates.parquet")
    p.add_argument("--output", type=Path,
                   default=Path("rasyn/data/clean/chembl_pair_rationale.parquet"))
    p.add_argument("--tier", default="silver",
                   help="Filter input to this quality_tier (default: silver).")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Process at most N pairs (debug / pilot).")
    p.add_argument("--bs", type=int, default=8,
                   help="Concurrent vLLM requests.")
    p.add_argument("--vllm-url", default=DEFAULT_VLLM)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="Write partial output every N pairs.")
    args = p.parse_args()

    if not PROMPT_PATH.exists():
        _log(f"FATAL: {PROMPT_PATH} not found")
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rasyn.papers.rationale_schemas import PairRationale

    prompt_md = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha = hash_text(prompt_md)
    PROMPT_LOCK.write_text(json.dumps({
        "sha256": prompt_sha,
        "n_chars": len(prompt_md),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
    }, indent=2))
    _log(f"Locked rationale prompt sha256: {prompt_sha[:16]}...")

    system_msg, user_template = parse_prompt(prompt_md)
    schema = PairRationale.model_json_schema()

    if not args.input.exists():
        _log(f"FATAL: input {args.input} not found")
        return 1

    df = pd.read_parquet(args.input)
    if args.tier and "quality_tier" in df.columns:
        before = len(df)
        df = df[df["quality_tier"] == args.tier]
        _log(f"Filtered to tier={args.tier}: {before:,} -> {len(df):,} pairs")
    if args.max_pairs:
        df = df.head(args.max_pairs)
        _log(f"Limited to first {args.max_pairs:,} pairs")
    if df.empty:
        _log("No pairs to process; exiting.")
        return 0

    # Resume if output exists
    done_ids: set[str] = set()
    if args.output.exists():
        prev = pd.read_parquet(args.output)
        done_ids = set(prev["pair_id"].astype(str).tolist())
        _log(f"Resuming: {len(done_ids):,} already in {args.output}")

    df = df[~df["pair_id"].astype(str).isin(done_ids)]
    if df.empty:
        _log("All pairs already processed.")
        return 0

    pairs = df.to_dict(orient="records")
    _log(f"Processing {len(pairs):,} new pairs (concurrency={args.bs})")

    results: list[dict] = []
    t0 = time.time()
    n_done = 0
    n_errors = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.bs) as pool:
        futures = {
            pool.submit(
                process_one_pair, pair,
                system_msg=system_msg, user_template=user_template, schema=schema,
                vllm_url=args.vllm_url, model_id=args.model_id, prompt_sha=prompt_sha,
            ): pair["pair_id"]
            for pair in pairs
        }
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            n_done += 1
            if res.get("error"):
                n_errors += 1
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                _log(f"  done {n_done:,}/{len(pairs):,} | err={n_errors} | {rate:.1f} pair/s")
            if n_done % args.checkpoint_every == 0:
                # Append to parquet checkpoint
                prev = pd.read_parquet(args.output) if args.output.exists() else pd.DataFrame()
                combined = pd.concat([prev, pd.DataFrame(results)], ignore_index=True, sort=False)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                combined.to_parquet(args.output, compression="zstd", index=False)
                results.clear()
                _log(f"  checkpoint -> {args.output} ({len(combined):,} rows)")

    if results:
        prev = pd.read_parquet(args.output) if args.output.exists() else pd.DataFrame()
        combined = pd.concat([prev, pd.DataFrame(results)], ignore_index=True, sort=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(args.output, compression="zstd", index=False)

    elapsed = time.time() - t0
    _log("")
    _log("=" * 60)
    _log(f"Total: {n_done} pairs | errors: {n_errors} | wall-clock: {elapsed/60:.1f} min")
    _log(f"Output: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
