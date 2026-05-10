"""P-0.5 driver: validate extraction prompt against ground-truth set.

For each GT paper with available text:
  1. Build system + user messages from locked extraction_prompt.md
  2. Call vLLM /v1/chat/completions with structured-output JSON schema
  3. Parse response into ExtractedRescuePairBatch
  4. Run extraction_validator.validate_batch() to filter
  5. Compare extracted pairs to GT expected pairs (match by InChIKey pair)

Aggregate:
  - True Positive: extracted pair matches a GT pair (parent + candidate InChIKey)
  - False Positive: extracted pair doesn't match any GT pair (hallucination)
  - False Negative: GT pair has no matching extracted pair (missed)
  - Precision = TP / (TP + FP)
  - Recall    = TP / (TP + FN)

Gate: precision >= 0.95 must be achieved before unlocking P-1 corpus run.

vLLM endpoint default = http://localhost:8000 (assumes SSH tunnel open from
local OR script running on vLLM pod itself):
  ssh -L 8000:localhost:8000 root@154.54.102.45 -p 18129 -i ~/.ssh/id_ed25519 -N

Run:
  python scripts/p0_5_validate_prompt.py
  # or with explicit endpoint:
  python scripts/p0_5_validate_prompt.py --vllm-url http://1.2.3.4:8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROMPT_PATH = Path("rasyn/papers/extraction_prompt.md")
GT_POPULATED = Path("rasyn/papers/ground_truth_set.populated.yaml")
TEXT_DIR = Path("rasyn/papers/gt_papers_text")
RESULTS_DIR = Path("rasyn/papers/p0_5_results")
FORBIDDEN_AUTHORS_PATH = Path("rasyn/papers/forbidden_authors.yaml")
PROMPT_LOCK = Path("rasyn/papers/extraction_prompt.lock.json")

DEFAULT_MODEL = "casperhansen/llama-3.3-70b-instruct-awq"
DEFAULT_VLLM_URL = "http://localhost:8000"


def _log(msg: str) -> None:
    print(msg, flush=True)


def hash_prompt(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def parse_prompt(md: str) -> tuple[str, str]:
    """Extract the system message + user-message template from the locked prompt MD.

    Looks for ```...``` blocks under the '## System message' and '## Per-paper
    user message (template)' headings. Returns (system, user_template).
    """
    sys_match = re.search(r"##\s+System message\s*\n+```\n(.*?)\n```", md, re.DOTALL)
    usr_match = re.search(
        r"##\s+Per-paper user message \(template\)\s*\n.*?```\n(.*?)\n```",
        md,
        re.DOTALL,
    )
    if not sys_match or not usr_match:
        raise SystemExit(
            "Failed to extract system/user template from extraction_prompt.md. "
            "The MD structure may have changed."
        )
    return sys_match.group(1), usr_match.group(1)


def call_vllm(
    *,
    vllm_url: str,
    model_id: str,
    system_msg: str,
    user_msg: str,
    json_schema: dict,
    timeout: int = 600,
) -> tuple[dict, int]:
    """Call vLLM /v1/chat/completions with structured-output. Returns (response_json, runtime_ms)."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ExtractedRescuePairBatch",
                "schema": json_schema,
                "strict": True,
            },
        },
        "temperature": 0.0,
        "max_tokens": 8192,
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


def gt_expected_pairs(gt_pair: dict) -> list[tuple[str | None, str | None]]:
    """Extract expected (parent_inchi_key, candidate_inchi_key) for matching."""
    p_ik = gt_pair["parent"].get("inchi_key")
    c_ik = gt_pair["candidate"].get("inchi_key")
    return [(p_ik, c_ik)] if p_ik and c_ik else []


def match_extraction(
    extracted_pairs: list,  # list of validated ExtractedRescuePair
    gt_pair: dict,
) -> dict:
    """Match extracted pairs against the GT pair. Returns counts dict."""
    expected_pairs = gt_expected_pairs(gt_pair)
    if not expected_pairs:
        # GT pair has no SMILES -> can't match; skip from precision/recall
        return {"tp": 0, "fp": 0, "fn": 0, "skipped": 1, "extracted_count": len(extracted_pairs)}

    # We don't have InChIKey on extracted pairs out of the box; the validator
    # populates it. Caller passes ValidationResult objects.
    expected_set = {(p.upper(), c.upper()) for p, c in expected_pairs}

    tp = 0
    fp = 0
    matched_expected: set[tuple[str, str]] = set()
    for r in extracted_pairs:
        if not r.accepted:
            continue
        p_ik = (r.parent_inchi_key or "").upper()
        c_ik = (r.candidate_inchi_key or "").upper()
        match = None
        for ep, ec in expected_set:
            # Either orientation matches (parent->candidate or candidate->parent)
            if (p_ik == ep and c_ik == ec) or (p_ik == ec and c_ik == ep):
                match = (ep, ec)
                break
        if match:
            if match not in matched_expected:
                tp += 1
                matched_expected.add(match)
            # if already matched, count as duplicate-tp (don't double-count)
        else:
            fp += 1
    fn = len(expected_set) - len(matched_expected)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "skipped": 0,
        "extracted_count": sum(1 for r in extracted_pairs if r.accepted),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vllm-url", default=DEFAULT_VLLM_URL,
                   help="vLLM endpoint base URL (default: localhost:8000).")
    p.add_argument("--model-id", default=DEFAULT_MODEL,
                   help="Model id to send to vLLM.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N GT papers (debug).")
    args = p.parse_args()

    # Lazy imports so the script's --help works without dependencies.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rasyn.papers.schemas import ExtractedRescuePairBatch
    from rasyn.papers.extraction_validator import (
        validate_batch,
        load_forbidden_authors_cfg,
    )

    if not PROMPT_PATH.exists():
        _log(f"FATAL: {PROMPT_PATH} not found")
        return 1
    if not GT_POPULATED.exists():
        _log(f"FATAL: {GT_POPULATED} not found. Run scripts/populate_gt_smiles.py first.")
        return 1

    prompt_md = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha = hash_prompt(prompt_md)
    PROMPT_LOCK.write_text(json.dumps({
        "sha256": prompt_sha,
        "n_chars": len(prompt_md),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
    }, indent=2))
    _log(f"Locked prompt sha256: {prompt_sha[:16]}...")

    system_msg, user_template = parse_prompt(prompt_md)
    json_schema = ExtractedRescuePairBatch.model_json_schema()

    gt = yaml.safe_load(GT_POPULATED.read_text())
    forbidden_cfg = load_forbidden_authors_cfg(FORBIDDEN_AUTHORS_PATH)

    # Sealed-case answers (placeholder; will be empty until registry populator runs)
    sealed_answers: list[tuple[str, str]] = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "tp": 0, "fp": 0, "fn": 0, "skipped": 0,
        "n_papers_processed": 0,
        "n_papers_skipped_missing_text": 0,
        "n_papers_skipped_llm_error": 0,
        "per_paper": [],
    }

    pairs_list = gt["pairs"]
    if args.limit:
        pairs_list = pairs_list[: args.limit]

    for gt_pair in pairs_list:
        gt_id = gt_pair["id"]
        doi = gt_pair.get("paper_doi_TODO") or gt_pair.get("paper_doi")
        if not doi:
            _log(f"[{gt_id}] no DOI; skipped")
            summary["n_papers_skipped_missing_text"] += 1
            continue

        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", doi)
        text_path = TEXT_DIR / f"{safe}.txt"
        if not text_path.exists():
            _log(f"[{gt_id}] text not fetched yet ({text_path}); run scripts/fetch_gt_papers.py")
            summary["n_papers_skipped_missing_text"] += 1
            continue

        paper_text = text_path.read_text(encoding="utf-8", errors="replace")
        # Truncate to keep within max-tokens window. Llama-3.3 8K context;
        # leave room for system + response (~3K). Cap text at ~5000 tokens
        # ~= 20K chars conservative.
        if len(paper_text) > 20_000:
            _log(f"[{gt_id}] truncating paper text {len(paper_text):,} -> 20,000 chars")
            paper_text = paper_text[:20_000]

        user_msg = (user_template
                    .replace("{{PAPER_DOI}}", doi)
                    .replace("{{PAPER_PMID}}", gt_pair.get("paper_pmid", "null") or "null")
                    .replace("{{PAPER_TITLE}}", gt_pair.get("title", ""))
                    .replace("{{PAPER_TEXT}}", paper_text))

        _log(f"[{gt_id}] calling vLLM ({args.model_id}) on {len(paper_text):,}-char text...")
        try:
            resp, runtime_ms = call_vllm(
                vllm_url=args.vllm_url, model_id=args.model_id,
                system_msg=system_msg, user_msg=user_msg, json_schema=json_schema,
            )
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            _log(f"  vLLM call failed: {e}")
            summary["n_papers_skipped_llm_error"] += 1
            continue

        try:
            content = resp["choices"][0]["message"]["content"]
            llm_obj = json.loads(content)
            llm_obj.setdefault("paper_doi", doi)
            llm_obj.setdefault("paper_pmid", gt_pair.get("paper_pmid"))
            llm_obj.setdefault("paper_title", gt_pair.get("title"))
            llm_obj.setdefault("extraction_timestamp_utc", datetime.now(timezone.utc).isoformat())
            llm_obj.setdefault("model_id", args.model_id)
            llm_obj.setdefault("prompt_sha256", prompt_sha)
            llm_obj["extraction_runtime_ms"] = runtime_ms
            batch = ExtractedRescuePairBatch.model_validate(llm_obj)
        except Exception as e:
            _log(f"  Failed to parse/validate LLM output: {e}")
            summary["n_papers_skipped_llm_error"] += 1
            continue

        report = validate_batch(
            batch,
            sealed_answer_smiles=sealed_answers,
            forbidden_authors_cfg=forbidden_cfg,
            paper_authors=None,  # CrossRef lookup deferred
        )

        # Save per-paper artefacts
        out_path = RESULTS_DIR / f"{safe}.json"
        out_path.write_text(json.dumps({
            "gt_id": gt_id,
            "doi": doi,
            "raw_response": llm_obj,
            "validation_report": report.to_dict(),
            "runtime_ms": runtime_ms,
        }, indent=2))

        match = match_extraction(report.per_pair, gt_pair)
        _log(f"  -> extracted {match['extracted_count']} validated, "
             f"TP={match['tp']} FP={match['fp']} FN={match['fn']}")

        summary["tp"] += match["tp"]
        summary["fp"] += match["fp"]
        summary["fn"] += match["fn"]
        summary["skipped"] += match["skipped"]
        summary["n_papers_processed"] += 1
        summary["per_paper"].append({
            "gt_id": gt_id,
            "doi": doi,
            "match": match,
            "n_extracted": report.n_pairs_input,
            "n_accepted": report.n_pairs_accepted,
            "drop_reasons": report.drop_reason_counts,
            "runtime_ms": runtime_ms,
        })

    tp = summary["tp"]
    fp = summary["fp"]
    fn = summary["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    summary["precision"] = precision
    summary["recall"] = recall

    (RESULTS_DIR / "_summary.json").write_text(json.dumps(summary, indent=2))

    _log("")
    _log("=" * 60)
    _log(f"Papers processed: {summary['n_papers_processed']}")
    _log(f"Papers skipped (missing text): {summary['n_papers_skipped_missing_text']}")
    _log(f"Papers skipped (LLM error):    {summary['n_papers_skipped_llm_error']}")
    _log(f"TP: {tp}   FP: {fp}   FN: {fn}   skipped-pairs: {summary['skipped']}")
    _log(f"Precision: {precision:.3f}   Recall: {recall:.3f}")
    _log(f"Gate (precision >= 0.95): {'PASS' if precision >= 0.95 else 'FAIL'}")
    _log("=" * 60)
    if precision < 0.95:
        _log("\nIterate the extraction_prompt.md (re-version), re-run, until precision passes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
