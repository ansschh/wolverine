"""Open Reaction Database (ORD) adapter (RETRO_PLAN R-1, MVP data pack item 4).

ORD is distributed as protobuf-encoded `Dataset` messages, optionally
gzipped. Each dataset is one of ~250 .pb.gz files in the
`open-reaction-database/ord-data` GitHub repo. We download the archive,
iterate datasets, and yield one reaction dict per `Reaction` message.

The ORD schema (open_reaction_database.schema.proto):
  Reaction
    .inputs[ReactionInput]      -> components -> ReactionInputCompound
    .outcomes[ReactionOutcome]  -> products[ProductCompound], conversion
    .conditions
      .pressure
      .temperature
      .stirring
      .illumination
    .notes
    .provenance
"""
from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

DEFAULT_ORD_DIR = Path("rasyn/data/raw/ord")
ORD_DATA_GIT = "https://github.com/open-reaction-database/ord-data.git"


@dataclass
class ORDConfig:
    raw_dir: Path = DEFAULT_ORD_DIR
    use_orderly_cleaned: bool = True  # prefer ORDerly-cleaned subset if available
    git_clone_depth: int = 1


def clone_ord_data(cfg: ORDConfig) -> Path:
    """Git-clone the ord-data repo (shallow). Idempotent."""
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    target = cfg.raw_dir / "ord-data"
    if target.exists():
        return target
    subprocess.check_call([
        "git", "clone", "--depth", str(cfg.git_clone_depth), ORD_DATA_GIT, str(target),
    ])
    return target


def _iter_pb_files(root: Path) -> Iterator[Path]:
    yield from sorted(root.rglob("*.pb.gz"))


def _stream_reactions_from_pb(path: Path) -> Iterator[dict]:
    """Stream one dict per Reaction message in a single ORD .pb.gz file.

    Requires the `ord-schema` Python package (pip install ord-schema).
    We import lazily so that the rest of this module can be inspected
    without it installed (e.g., in CI on a machine without ord-schema).
    """
    try:
        from ord_schema.proto import dataset_pb2  # type: ignore[import-not-found]
    except ImportError as e:  # noqa: BLE001 -- explicit re-raise with context
        raise RuntimeError(
            "ord-schema not installed; pip install ord-schema to ingest ORD"
        ) from e

    with gzip.open(path, "rb") as fh:
        dataset = dataset_pb2.Dataset()
        dataset.ParseFromString(fh.read())

    for rxn in dataset.reactions:
        reactants: list[str] = []
        reagents: list[str] = []
        solvents: list[str] = []
        catalysts: list[str] = []
        for input_name, rxn_input in rxn.inputs.items():
            for comp in rxn_input.components:
                ident_smiles = None
                for ident in comp.identifiers:
                    if ident.type == ident.SMILES:
                        ident_smiles = ident.value
                        break
                if ident_smiles is None:
                    continue
                role = getattr(comp, "reaction_role", None)
                # Roles: REACTANT=1, REAGENT=2, SOLVENT=3, CATALYST=4, ...
                if role == 3:
                    solvents.append(ident_smiles)
                elif role == 4:
                    catalysts.append(ident_smiles)
                elif role == 2:
                    reagents.append(ident_smiles)
                else:
                    reactants.append(ident_smiles)

        product_smiles = None
        yield_pct = None
        for outcome in rxn.outcomes:
            for p in outcome.products:
                for ident in p.identifiers:
                    if ident.type == ident.SMILES:
                        product_smiles = ident.value
                        break
                if product_smiles is not None:
                    # Try to extract yield from product measurements
                    for m in p.measurements:
                        if m.type == m.YIELD:
                            try:
                                yield_pct = float(m.percentage.value)
                            except Exception:
                                pass
                            break
                    break
            if product_smiles is not None:
                break

        if not product_smiles or not reactants:
            continue

        temperature_c = None
        if rxn.HasField("conditions"):
            cond = rxn.conditions
            if cond.HasField("temperature"):
                t = cond.temperature
                if t.HasField("setpoint"):
                    sp = t.setpoint
                    val = sp.value
                    unit = sp.units
                    # Units: KELVIN=1, CELSIUS=2, FAHRENHEIT=3
                    if unit == 1:
                        temperature_c = val - 273.15
                    elif unit == 2:
                        temperature_c = val
                    elif unit == 3:
                        temperature_c = (val - 32) * 5 / 9

        doc = None
        if rxn.HasField("provenance"):
            prov = rxn.provenance
            if prov.HasField("publication_url"):
                doc = prov.publication_url.value if hasattr(prov.publication_url, "value") else prov.publication_url
            elif hasattr(prov, "doi") and prov.doi:
                doc = prov.doi
            elif hasattr(prov, "patent") and prov.patent:
                doc = prov.patent

        yield {
            "source": "ord",
            "source_record_id": rxn.reaction_id or None,
            "reactants": reactants,
            "reagents": reagents,
            "solvents": solvents,
            "catalysts": catalysts,
            "product": product_smiles,
            "rxn_smiles": ".".join(reactants) + ">" + ".".join(reagents) + ">" + product_smiles,
            "yield_pct": yield_pct,
            "temperature_c": temperature_c,
            "document_id": doc,
        }


def stream_ord(cfg: ORDConfig) -> Iterator[dict]:
    """Top-level: clone if needed, then iterate every Reaction in every dataset."""
    root = clone_ord_data(cfg)
    for pb in _iter_pb_files(root):
        try:
            yield from _stream_reactions_from_pb(pb)
        except Exception as e:  # noqa: BLE001 -- per-dataset isolation
            print(f"[ord] failed to parse {pb}: {e}", file=sys.stderr)
            continue


def stream_orderly_cleaned(orderly_parquet_path: Path) -> Iterator[dict]:
    """If ORDerly-cleaned parquet is available locally, prefer it.

    ORDerly publishes pre-cleaned ORD subsets with normalized fields:
    https://github.com/sustainable-processes/ORDerly
    """
    import pyarrow.parquet as pq  # local import to keep top-level imports light
    table = pq.read_table(orderly_parquet_path)
    rows = table.to_pylist()
    for row in rows:
        product = row.get("product_000") or row.get("product")
        if not product:
            continue
        reactants = [v for k, v in row.items() if k.startswith("reactant_") and v]
        reagents = [v for k, v in row.items() if k.startswith("reagent_") and v]
        solvents = [v for k, v in row.items() if k.startswith("solvent_") and v]
        catalysts = [v for k, v in row.items() if k.startswith("agent_") and v]
        yield {
            "source": "ord_erly",
            "source_record_id": row.get("rxn_id") or row.get("reaction_id"),
            "reactants": reactants,
            "reagents": reagents,
            "solvents": solvents,
            "catalysts": catalysts,
            "product": product,
            "rxn_smiles": ".".join(reactants) + ">" + ".".join(reagents) + ">" + product,
            "yield_pct": row.get("yield_000") or row.get("yield"),
            "temperature_c": row.get("temperature"),
            "document_id": row.get("doi") or row.get("patent"),
        }
