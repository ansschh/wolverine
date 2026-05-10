"""Pydantic schemas for paper/patent SAR extraction (P-0 deliverable).

ExtractedRescuePair is the structured output we ask the LLM to produce per
rescue case found in a paper. ExtractedRescuePairBatch wraps the LLM's
response for ONE paper (zero or more pairs, plus extraction provenance).

The schema is the contract:
- LLM is prompted with a JSON Schema generated from these models
  (vLLM xgrammar / response_format).
- Post-LLM validation in `extraction_validator.py` runs RDKit SMILES
  round-trip, decontamination check, ChEMBL cross-ref, forbidden-author
  check. Pairs that fail any validator are dropped (logged with reason).

Per L25 (HARD): no fallbacks. If a pair fails to satisfy any required field,
the LLM must omit it (return empty `valid_pairs` for the paper if no
satisfying pair exists). Per L33: never pad.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Liability families locked at v1 per spec rasyn_curating_the_dataset.md §6.
LiabilityType = Literal[
    "hERG",
    "solubility",
    "metabolic_stability",
    "oral_exposure",
    "permeability",
]

# Confidence is the LLM's own self-assessment of extraction certainty.
ExtractionConfidence = Literal["high", "medium", "low"]


class ExtractedRescuePair(BaseModel):
    """One parent->candidate rescue pair extracted from a paper.

    A pair is valid only if ALL of:
      1. parent_smiles + candidate_smiles are explicit in the paper (SI table,
         InChI, named CAS/IUPAC), NOT inferred from drawings alone.
      2. Both compounds were measured in the SAME assay under the SAME
         conditions for the liability_type endpoint.
      3. The paper narrative explicitly describes the candidate as an
         improvement in the liability (not inferred from structure).
      4. Primary-target potency is preserved within 10x or paper explicitly
         says "retained activity" (captured in retention_check_metric).

    These rules are baked into the extraction prompt; this schema enforces
    that the LLM must structure its output to expose them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ----- Provenance -----
    paper_doi: str = Field(description="Paper DOI (e.g. '10.1021/jm060379l').")
    paper_pmid: str | None = Field(
        default=None, description="PubMed ID if available (e.g. '17048954')."
    )

    # ----- Parent compound -----
    parent_smiles: str = Field(
        description="Parent compound SMILES exactly as found in paper SI/InChI."
    )
    parent_name_in_paper: str = Field(
        description="Parent compound name verbatim from the paper text/table "
                    "(e.g. 'compound 4a', 'cisapride', 'Pf-2341066')."
    )
    parent_table_ref: str | None = Field(
        default=None,
        description="Where in paper this compound appears, e.g. "
                    "'Table 2 entry 4a', 'Figure 3 compound 12', 'SI section S2.3'."
    )

    # ----- Candidate (rescue) compound -----
    candidate_smiles: str = Field(
        description="Candidate compound SMILES exactly as found in paper SI/InChI."
    )
    candidate_name_in_paper: str = Field(
        description="Candidate compound name verbatim from paper."
    )
    candidate_table_ref: str | None = Field(
        default=None, description="Same convention as parent_table_ref."
    )

    # ----- Liability rescue context -----
    liability_type: LiabilityType = Field(
        description="Which ADMET liability is rescued."
    )
    liability_endpoint: str = Field(
        description="Specific assay endpoint as stated in paper, e.g. "
                    "'hERG IC50', 'Caco-2 Papp e6 cm/s', 'logS', "
                    "'Half-life Obach', 'F% bioavailability rat'."
    )

    # ----- Measured values -----
    parent_metric: dict[str, float] = Field(
        description="Parent's measurement(s) for the liability assay. "
                    "Keys are descriptive (e.g. 'hERG_IC50_uM'), values "
                    "numeric. At minimum the endpoint value must be present."
    )
    candidate_metric: dict[str, float] = Field(
        description="Candidate's measurement(s) for the SAME liability assay."
    )
    retention_check_metric: dict[str, float] | None = Field(
        default=None,
        description="Primary-target potency for both compounds, used to verify "
                    "activity was retained. Keys 'parent_*' / 'candidate_*'. "
                    "If paper only states 'retained activity' qualitatively, "
                    "set to None and add a warning."
    )

    # ----- Activity context -----
    target: str | None = Field(
        default=None, description="Primary therapeutic target name (e.g. 'EGFR')."
    )
    target_chembl_id: str | None = Field(
        default=None, description="ChEMBL target ID if known."
    )

    # ----- Transformation -----
    transformation_class: str | None = Field(
        default=None,
        description="Medicinal-chemistry transformation class, e.g. "
                    "'prodrug_ester', 'bioisostere_phenyl_to_pyridyl', "
                    "'soft_spot_block', 'basicity_reduction', 'polar_addition'."
    )

    # ----- Confidence + warnings -----
    confidence: ExtractionConfidence = Field(
        description="LLM's confidence in this extraction. 'high' means SMILES "
                    "from SI/InChI + measured values explicit; 'medium' means "
                    "values from text but SMILES from named compound lookup; "
                    "'low' means partial info, needs review (drop in P-2)."
    )
    extraction_warnings: list[str] = Field(
        default_factory=list,
        description="List of caveats noticed during extraction. Examples: "
                    "'SMILES inferred from compound name in text', "
                    "'parent and candidate measured in different labs'."
    )

    # ----- Validators -----
    @field_validator("parent_smiles", "candidate_smiles")
    @classmethod
    def _smiles_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("SMILES must be non-empty.")
        return v.strip()

    @field_validator("parent_metric", "candidate_metric")
    @classmethod
    def _metric_nonempty(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("metric dict must contain at least one endpoint value.")
        return v

    @model_validator(mode="after")
    def _no_self_pair(self):
        if self.parent_smiles == self.candidate_smiles:
            raise ValueError(
                f"parent_smiles and candidate_smiles are identical: {self.parent_smiles}"
            )
        if not self.paper_doi.startswith("10."):
            raise ValueError(f"paper_doi must look like a DOI ('10.xxx/yyy'); got {self.paper_doi!r}")
        return self


class ExtractedRescuePairBatch(BaseModel):
    """LLM's response for ONE paper.

    Empty `valid_pairs` is a valid response — paper had no rescue pairs
    satisfying the 4 hard rules (per L33: never pad). The full provenance
    block lets us reproduce + audit each extraction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    paper_doi: str = Field(description="DOI of the paper this batch was extracted from.")
    paper_pmid: str | None = None
    paper_title: str | None = None

    valid_pairs: list[ExtractedRescuePair] = Field(
        default_factory=list,
        description="Zero or more rescue pairs satisfying all 4 hard rules. "
                    "Per L33: do NOT pad if no satisfying pair exists."
    )

    # ----- Extraction provenance -----
    extraction_timestamp_utc: str = Field(
        description="ISO 8601 UTC timestamp when LLM produced this output."
    )
    model_id: str = Field(
        description="Serving model id, e.g. 'casperhansen/llama-3.3-70b-instruct-awq'."
    )
    prompt_sha256: str = Field(
        description="SHA256 hex of the locked extraction_prompt.md content."
    )
    extraction_runtime_ms: int | None = Field(
        default=None, description="Wall-clock LLM runtime in ms."
    )

    @model_validator(mode="after")
    def _per_pair_doi_match(self):
        for p in self.valid_pairs:
            if p.paper_doi != self.paper_doi:
                raise ValueError(
                    f"valid_pair.paper_doi {p.paper_doi!r} != batch.paper_doi {self.paper_doi!r}"
                )
        return self
