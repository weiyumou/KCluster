# Spacing study

Driver for the spacing-study transaction exports (Podsie; DataShop ds6527-era
templates). `processing.py` cleans a raw transaction file (column selection,
stage filter, stable question IDs derived from problem text + answer options)
and extracts per-course question JSONL for all four question types;
`create_ds_kc.py` populates the DataShop KC models (defaults, Standard,
Q-Group, plus the KCluster/Concept results).

Run from this directory, e.g.:

    python processing.py --raw_tx_path raw_data/<export>.txt --output_dir data/

This pipeline supersedes the older `kcluster/datasets/spacing.py` extractor
from the legacy main repo (pre-cleaning export schema, `Input`-grouped
choices), which was not ported.
