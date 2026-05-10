-- ChEMBL doc-layer filter: candidate papers for ADMET-rescue extraction.
--
-- Selects ChEMBL `docs` rows where the document:
--   1. Is a published paper (not patent or unpublished) with a DOI
--   2. Has >= 10 distinct molecules tested
--   3. Has at least one assay tagged for an ADMET liability of interest
--      (hERG, solubility, metabolic stability, oral exposure, permeability)
--   4. Is NOT in our quarantine list (L10 + forbidden_authors.yaml DOIs)
--
-- This is the primary corpus filter for P-1 extraction. Output is written
-- to candidate_papers.parquet and triaged before LLM calls.
--
-- Typical yield from ChEMBL 35: ~150-400 papers per liability x ~5 liabilities
-- ≈ ~500-1500 candidates BEFORE triage. After title-pattern triage we expect
-- ~100-500 papers per L33 quality target.
--
-- Run:
--   sqlite3 chembl_35.db < source_filter.sql > candidate_papers.tsv
--
-- Columns: doc_chembl_id, doi, pubmed_id, year, journal, title,
--          n_molecules, n_admet_assays, liability_types_seen
--
-- Rerun is cheap (no GPU); safe to iterate the WHERE clauses.

-- Step 1: standard_type -> liability_type mapping (matches
--         CHEMBL_STDTYPE_TO_LIABILITY in build_rescue_pair_dataset.py).
WITH liability_stdtypes AS (
    SELECT 'hERG IC50' AS standard_type, 'hERG' AS liability UNION ALL
    SELECT 'IC50',                        'hERG'           UNION ALL  -- only when target is CHEMBL240
    SELECT 'Solubility',                  'solubility'    UNION ALL
    SELECT 'logS',                        'solubility'    UNION ALL
    SELECT 'Aqueous solubility',          'solubility'    UNION ALL
    SELECT 'Cl',                          'metabolic_stability' UNION ALL
    SELECT 'CL',                          'metabolic_stability' UNION ALL
    SELECT 'Half life',                   'metabolic_stability' UNION ALL
    SELECT 'Half-life',                   'metabolic_stability' UNION ALL
    SELECT 'T1/2',                        'metabolic_stability' UNION ALL
    SELECT 'Stability',                   'metabolic_stability' UNION ALL
    SELECT 'Bioavailability',             'oral_exposure' UNION ALL
    SELECT 'F',                           'oral_exposure' UNION ALL
    SELECT 'AUC',                         'oral_exposure' UNION ALL
    SELECT 'Caco-2',                      'permeability'  UNION ALL
    SELECT 'PAMPA',                       'permeability'  UNION ALL
    SELECT 'Papp',                        'permeability'
),

-- Step 2: per-doc per-liability assay counts.
doc_admet AS (
    SELECT
        d.doc_id,
        d.chembl_id     AS doc_chembl_id,
        d.doi,
        d.pubmed_id,
        d.year,
        d.journal,
        d.title,
        ls.liability   AS liability_type,
        COUNT(DISTINCT act.molregno) AS n_molecules_in_liability,
        COUNT(DISTINCT a.assay_id)   AS n_admet_assays_in_liability
    FROM docs d
    JOIN activities act ON act.doc_id = d.doc_id
    JOIN assays a       ON a.assay_id = act.assay_id
    JOIN liability_stdtypes ls
        ON ls.standard_type = act.standard_type
        OR (ls.liability = 'hERG' AND act.standard_type = 'IC50'
            AND a.tid IN (SELECT tid FROM target_dictionary
                           WHERE chembl_id = 'CHEMBL240'))
    WHERE d.doi IS NOT NULL
      AND d.doc_type = 'PUBLICATION'           -- exclude patents at this filter
      AND act.standard_value IS NOT NULL
      AND act.standard_relation = '='
    GROUP BY d.doc_id, d.chembl_id, d.doi, d.pubmed_id,
             d.year, d.journal, d.title, ls.liability
),

-- Step 3: per-doc total distinct molecule count (across ALL assays).
doc_molcount AS (
    SELECT
        d.doc_id,
        COUNT(DISTINCT act.molregno) AS n_distinct_molecules
    FROM docs d
    JOIN activities act ON act.doc_id = d.doc_id
    GROUP BY d.doc_id
),

-- Step 4: aggregate across liabilities for each doc.
doc_summary AS (
    SELECT
        da.doc_chembl_id,
        da.doi,
        da.pubmed_id,
        da.year,
        da.journal,
        da.title,
        dmc.n_distinct_molecules,
        SUM(da.n_admet_assays_in_liability) AS n_admet_assays,
        GROUP_CONCAT(DISTINCT da.liability_type) AS liability_types_seen
    FROM doc_admet da
    JOIN doc_molcount dmc ON dmc.doc_id = (
        SELECT d.doc_id FROM docs d WHERE d.chembl_id = da.doc_chembl_id
    )
    GROUP BY da.doc_chembl_id, da.doi, da.pubmed_id, da.year,
             da.journal, da.title, dmc.n_distinct_molecules
)

-- Step 5: filter + sort. Forbidden DOIs (L10) are excluded here; forbidden-
--         author exclusion happens in Python (extraction_validator.py)
--         since it requires CrossRef API metadata.
SELECT
    doc_chembl_id,
    doi,
    pubmed_id,
    year,
    journal,
    title,
    n_distinct_molecules,
    n_admet_assays,
    liability_types_seen
FROM doc_summary
WHERE n_distinct_molecules >= 10
  AND n_admet_assays >= 1
  AND year >= 2000              -- modern assay protocols only
  AND doi NOT IN (
    -- L10: ADMET-003 forbidden DOI
    '10.1039/d4md00275j'
    -- additional forbidden DOIs go here as discovered
  )
ORDER BY n_admet_assays DESC, n_distinct_molecules DESC;
