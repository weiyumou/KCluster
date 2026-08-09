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

## Format balance, and why it matters here

The 13 courses hold 7,069 questions, balanced almost exactly four ways by
question format (1,767 multiple choice / 1,766 short answer / 1,768 true-false /
1,768 fill-in). That balance is deliberate in the study design, and it makes thisthe worst case for format leakage in question congruity: pairs score congruent
partly for sharing a format, and affinity propagation then recovers format
families rather than KCs. Cluster with `--residualize` (decision D9) and compare
against the uncorrected model rather than assuming either is right.

Job scripts that ran KCluster on this workspace live in the git-ignored `jobs/`
(see `datasets/README.md`); `jobs/vertex-kcluster.sh` drives the Vertex AI path,
one batch job per course.
