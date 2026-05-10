"""Paper / patent SAR extraction workstream (P-0 .. P-5 per PLAN.md §16).

This subpackage is the deferred methodology-first paper-extraction pipeline:
- schemas.py            ExtractedRescuePair Pydantic v2 model + batch wrapper
- extraction_validator  Deterministic post-LLM validation (SMILES round-trip,
                        decontamination, ChEMBL cross-ref, forbidden-author
                        check)
- ground_truth_set.yaml 15 textbook rescue pairs for prompt validation (P-0.5)
- forbidden_authors.yaml Author quarantine extending L10 DOI quarantine
- extraction_prompt.md  Locked single-paper extraction prompt (SHA256-hashed)
- source_filter.sql     ChEMBL doc-layer filter producing candidate paper list

Per L33 (HARD): quality > quantity. Target ~100-500 carefully-selected papers
producing ~500-2K gold rescue pairs. NOT mass extraction.
Per L25 (HARD): no fallbacks/placeholders/shortcuts.
Per L16: gold tier (paper-curated rescue pairs) gated on full P-0..P-5
methodology lock + dry-run validation.
"""
