# Dataset workspaces

One directory per dataset KCluster has been applied to. Each workspace holds
the course-specific *driver* scripts — the glue that turns one particular
course export into the Question JSONL format — while the reusable *format
adapters* they call (ScienceQA JSON, OLI HTML, DataShop templates) live in
`kcluster/io/loaders/` and ship with the package. Workspaces are tracked in
the repo but are **never** part of the `kcluster` wheel.

Conventions:

- Drivers import from the installed `kcluster` package (`pip install -e .`).
- Raw course exports go in `<name>/raw_data/`, produced question files in
  `<name>/data/` — both are git-ignored. **Never commit data** (student data,
  DataShop exports, course-derived question banks, images, run outputs); the
  pre-commit guard enforces this.
