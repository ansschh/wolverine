"""PubChem PUG REST + BioAssay FTP adapter.

PUG REST (https://pubchem.ncbi.nlm.nih.gov/rest/pug/) is used for targeted
lookups. BioAssay bulk lives on FTP (ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/)
and is too large for full ingestion at v1 — we subset to assays relevant
to our liability families (hERG, solubility, metabolic stability, oral exposure).

Rate limit: PUG REST ~5 req/sec (PubChem documentation). The lookup helpers
here do single requests per call; batch jobs should sleep between calls.
"""

from __future__ import annotations

import json
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
USER_AGENT = "rasyn/0.1 (research; +https://github.com/ansschh/wolverine)"


def _http_json(url: str, *, timeout: int = 30, retries: int = 3, sleep: float = 0.25) -> dict | None:
    """GET URL and parse JSON. Retries on transient failure."""
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(sleep * (2**attempt))
    return None


def cids_by_name(name: str) -> list[int]:
    """All PubChem CIDs matching a compound name (synonym lookup)."""
    url = f"{PUG_REST}/compound/name/{quote(name)}/cids/JSON"
    data = _http_json(url)
    if not data:
        return []
    return list((data.get("IdentifierList") or {}).get("CID", []))


def properties_by_cid(cid: int, props: tuple[str, ...] = ("CanonicalSMILES", "InChIKey", "IUPACName", "MolecularFormula")) -> dict | None:
    """Fetch a property block for a single CID."""
    url = f"{PUG_REST}/compound/cid/{cid}/property/{','.join(props)}/JSON"
    data = _http_json(url)
    if not data:
        return None
    table = (data.get("PropertyTable") or {}).get("Properties") or []
    return table[0] if table else None


def synonyms_by_cid(cid: int) -> list[str]:
    """All synonyms for a single CID."""
    url = f"{PUG_REST}/compound/cid/{cid}/synonyms/JSON"
    data = _http_json(url)
    if not data:
        return []
    info = (data.get("InformationList") or {}).get("Information") or []
    return list(info[0].get("Synonym", [])) if info else []


def lookup_compound(name: str, *, prefer_first_cid: bool = True) -> dict | None:
    """End-to-end name -> {cid, smiles, inchi_key, iupac, synonyms}.

    With `prefer_first_cid=True` we take CID[0] (PubChem's preferred match);
    callers should verify before promoting to the sealed-case registry.
    """
    cids = cids_by_name(name)
    if not cids:
        return None
    cid = cids[0] if prefer_first_cid else min(cids)
    props = properties_by_cid(cid)
    if not props:
        return None
    return {
        "name": name,
        "pubchem_cid": str(cid),
        "candidate_cids": [str(c) for c in cids[:5]],
        "canonical_smiles": props.get("CanonicalSMILES"),
        "inchi_key": props.get("InChIKey"),
        "iupac_name": props.get("IUPACName"),
        "molecular_formula": props.get("MolecularFormula"),
        "synonyms": synonyms_by_cid(cid)[:50],
    }


# BioAssay liability-relevant AID hints (seed set; populator extends).
LIABILITY_AID_HINTS = {
    "hERG": ["1903", "588834"],
    "solubility": ["1996", "603846"],
    "metabolic_stability": ["1645841", "1645842"],
}
