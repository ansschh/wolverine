# rasyn

Implementation for the Rasyn ADMET rescue discovery system.

**This is the implementation tree.** The plan, specs, and running log live one directory up:

- [`../PLAN.md`](../PLAN.md) — master plan, read first
- [`../MEMORY.md`](../MEMORY.md) — running log, read second
- `../*.md` — the 5 source specs (locked, do not modify)

## Quick start

```bash
pip install -e ".[dev,chem]"
pytest tests/
```

## Layout

```
rasyn/                  # the importable package
  schemas/              # Pydantic v2 frozen schemas (the project contract)
  data/
    registry/           # sealed_case_registry.yaml, forbidden_entities.json, canaries.yaml
    raw/                # downloaded source data (gitignored)
    clean/              # 4 parquet tables + manifests + audits (gitignored except manifests)
  proposer/             # 6 proposer channels
  evidence/             # evidence builder (descriptors, similarity, deltas)
  aux_models/           # clean ADMET / activity-retention auxiliary predictors
  ranker/               # pairwise rescue ranker
  baselines/            # 8 baselines
  eval/                 # harness, metrics, functional-recovery scorer
  training/             # pretrain / train / finetune / calibrate
  audit/                # canary, NN audit, manifest, locked-prediction ledger
  reports/              # technical appendix, investor slide, per-case cards
tests/                  # tests + fixtures
```

## Schema-first

All cross-component contracts live in `rasyn/schemas/`. Every artifact (config, dataset manifest, model checkpoint, locked prediction) is hashed via canonical-JSON SHA256 (`rasyn.schemas.hashing.hash_model`).

If a schema and a spec disagree, the spec wins — update the schema and log the change in `../MEMORY.md`.
