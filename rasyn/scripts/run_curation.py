"""Run the 12-pass curation pipeline (Phase A-4).

Currently a skeleton; each pass logs its status to `curate_log.json`. Real
work activates as each pass's data lands.

    python scripts/run_curation.py
"""

from rasyn.data.curate.passes import run_all_passes

if __name__ == "__main__":
    out = run_all_passes()
    print(f"curate log: {out}")
