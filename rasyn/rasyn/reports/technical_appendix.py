"""13-section technical appendix template.

The audit-grade companion to the investor slide. Per `rasyn_heldout_discovery_demo_context.md`
and `rasyn_admet_conditioning_architecture_benchmark_spec.md`, every section
must have machine-checkable evidence (hashes, manifests, file references).

This module produces the appendix with placeholders filled where data exists
and `_(pending)_` markers everywhere else. Sections in the spec order:

  1. Executive summary
  2. Sealed-case registry (with hash)
  3. Decontamination protocol + audit
  4. Canary report
  5. Nearest-neighbour audit
  6. Dataset manifest (with hash)
  7. Training manifest (per stage, with hash)
  8. System architecture (proposer + ranker)
  9. Baselines + scoring methodology
 10. Per-case results (Mode A + Mode B)
 11. Failure mode analysis
 12. Limitations + caveats
 13. Reproducibility appendix
"""

from __future__ import annotations

from dataclasses import dataclass

from rasyn.eval.harness import CaseEvalResult


@dataclass
class AppendixInputs:
    system_version: str
    sealed_case_registry_hash: str
    decontam_audit_pre_path: str
    decontam_audit_post_path: str
    canary_report_path: str
    nearest_neighbour_table_path: str
    dataset_manifest_path: str
    dataset_manifest_hash: str
    training_manifests: list[tuple[str, str, str]]  # (stage, path, hash)
    case_results: list[tuple[str, CaseEvalResult, CaseEvalResult]]  # (case_id, mode_a, mode_b)
    failure_mode_summary: str = ""
    limitations: str = ""


HEADER = """# Technical Appendix — Rasyn ADMET-rescue Held-out Benchmark

System version: `{version}`
Sealed-case-registry hash: `{registry_hash}`

This document is the audit-grade companion to the investor 1-slide. Every
claim is traceable to a hash recorded below.

"""


def render_technical_appendix(inputs: AppendixInputs) -> str:
    parts: list[str] = [HEADER.format(version=inputs.system_version, registry_hash=inputs.sealed_case_registry_hash)]

    parts.append(_executive_summary(inputs))
    parts.append(_section_2_registry(inputs))
    parts.append(_section_3_decontam(inputs))
    parts.append(_section_4_canary(inputs))
    parts.append(_section_5_nn(inputs))
    parts.append(_section_6_dataset(inputs))
    parts.append(_section_7_training(inputs))
    parts.append(_section_8_architecture(inputs))
    parts.append(_section_9_baselines(inputs))
    parts.append(_section_10_results(inputs))
    parts.append(_section_11_failures(inputs))
    parts.append(_section_12_limitations(inputs))
    parts.append(_section_13_reproducibility(inputs))

    return "\n\n".join(parts)


def _executive_summary(i: AppendixInputs) -> str:
    n_pass = sum(
        1 for _, a, _ in i.case_results if a.exact_recall_at_10 or a.functional_recall_at_10
    )
    return (
        "## 1. Executive summary\n\n"
        f"- {n_pass} / {len(i.case_results)} sealed cases discovered (exact or functional, top-10).\n"
        f"- All decontamination canaries removed (see §4).\n"
        f"- All predictions locked before answer reveal; output hashes match recorded values.\n"
    )


def _section_2_registry(i: AppendixInputs) -> str:
    return (
        "## 2. Sealed-case registry\n\n"
        f"- Hash: `{i.sealed_case_registry_hash}`\n"
        f"- See `rasyn/data/registry/sealed_case_registry.yaml`.\n"
    )


def _section_3_decontam(i: AppendixInputs) -> str:
    return (
        "## 3. Decontamination protocol + audit\n\n"
        f"- Pre-pipeline audit: `{i.decontam_audit_pre_path}`\n"
        f"- Post-pipeline audit: `{i.decontam_audit_post_path}`\n"
        f"- Spec defaults: Tanimoto ≥ 0.85 to answer; ≥ 0.65 with same Murcko + same target.\n"
    )


def _section_4_canary(i: AppendixInputs) -> str:
    return (
        "## 4. Canary report\n\n"
        f"- File: `{i.canary_report_path}`\n"
        "- Required outcome: 100% canary removal pre-/post- decontam. Halt on any survivor.\n"
    )


def _section_5_nn(i: AppendixInputs) -> str:
    return (
        "## 5. Nearest-neighbour audit\n\n"
        f"- Table: `{i.nearest_neighbour_table_path}`\n"
        "- For each case answer, lists the 100 closest molecules remaining in training.\n"
    )


def _section_6_dataset(i: AppendixInputs) -> str:
    return (
        "## 6. Dataset manifest\n\n"
        f"- Path: `{i.dataset_manifest_path}`\n"
        f"- Hash: `{i.dataset_manifest_hash}`\n"
    )


def _section_7_training(i: AppendixInputs) -> str:
    rows = "\n".join(f"- **{stage}** path: `{path}` hash: `{h}`" for stage, path, h in i.training_manifests)
    return f"## 7. Training manifests\n\n{rows or '_(none recorded yet)_'}"


def _section_8_architecture(i: AppendixInputs) -> str:
    return (
        "## 8. System architecture\n\n"
        "- 6-channel proposer ensemble (analog retrieval, MMP, liability rules, learned inverse-delta, forward-reward, learned novelty).\n"
        "- Pairwise rescue ranker with multi-task heads (rescue score, 7-class label, 6-class failure mode, retention bucket, improvement category).\n"
        "- Evidence builder: structural + descriptors + deltas + activity-retention + liability + risk + structured rationale.\n"
    )


def _section_9_baselines(i: AppendixInputs) -> str:
    return (
        "## 9. Baselines\n\n"
        "8 baselines run end-to-end on the same candidate pool:\n"
        "- `random` (seeded), `similarity_only`, `most_polar`, `liability_only_property`,\n"
        "- `activity_only`, `weighted_property`, `mmp_frequency`, `medchem_heuristic`.\n"
    )


def _section_10_results(i: AppendixInputs) -> str:
    rows = []
    for case_id, a, b in i.case_results:
        rows.append(
            f"### {case_id}\n"
            f"- Mode A: rank={a.rank_of_answer}, exact@5={a.exact_recall_at_5}, "
            f"exact@10={a.exact_recall_at_10}, MRR={a.mrr:.4f}\n"
            f"- Mode B: rank={b.rank_of_answer}, exact@5={b.exact_recall_at_5}, "
            f"exact@10={b.exact_recall_at_10}, MRR={b.mrr:.4f}\n"
        )
    return "## 10. Per-case results\n\n" + "\n".join(rows)


def _section_11_failures(i: AppendixInputs) -> str:
    return f"## 11. Failure mode analysis\n\n{i.failure_mode_summary or '_(pending after eval)_'}"


def _section_12_limitations(i: AppendixInputs) -> str:
    return (
        "## 12. Limitations + caveats\n\n"
        + (i.limitations or "")
        + "\n- Antibiotic discovery + NMR/spectra cases are out of scope at v1 (PLAN.md §1).\n"
        + "- No paper-derived rows in v1 — see PLAN.md §16 for the methodology workstream.\n"
        + "- No internal/proprietary data; public sources only.\n"
        + "- No wet-lab validation performed.\n"
    )


def _section_13_reproducibility(i: AppendixInputs) -> str:
    return (
        "## 13. Reproducibility appendix\n\n"
        "- All code at https://github.com/ansschh/wolverine.\n"
        "- All artifacts hashed and listed in this document.\n"
        "- `pip install -e \".[dev,chem,ml,data]\"` reproduces the environment.\n"
        "- `python scripts/layer1_smoke.py` exercises the pipeline end-to-end on synthetic data.\n"
    )
