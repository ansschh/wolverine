"""Phase A-4: 12-pass curation pipeline orchestrator.

Each pass is a separate function so it can be re-run in isolation.
The driver `run_all_passes` chains them and writes intermediate outputs
that downstream passes consume.
"""
