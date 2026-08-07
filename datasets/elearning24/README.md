# E-learning 2024 (OLI Torus)

Driver for the 2024 offering of E-Learning Design Principles and Methods,
exported from OLI Torus as per-activity JSON. `mcq.py` extracts
multiple-choice questions, `mfb.py` extracts multi-input ("fill in the
blank") questions as one MCQ per blank, both joining against a DataShop
unique-step template and carrying learning objectives, feedback, and
images; `create_ds_kc.py` populates the DataShop KC models.

Extra dependencies (not part of the `kcluster` package): `requests` and
`pillow`, installed via the repo's `datasets` dependency group
(`uv sync` includes it through the dev group).

Run from this directory, e.g.:

    python mcq.py --step_path datashop/<unique-step>.txt --raw_data_dir raw_data/
