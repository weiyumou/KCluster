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
- No export is stored in this repository, not even ignored: every driver takes
  its input paths as arguments, so keep the data outside the working tree and
  pass the path (or symlink `<name>/raw_data` at it).
- Scripts that *ran* KCluster on a workspace go in `<name>/jobs/`, git-ignored
  alongside `data/` and `raw_data/`. A driver is reproducible and belongs in
  the repo; a job is one execution on one backend, carrying cluster paths and
  GCP identifiers that are local by nature — the same reason `vertex.toml` is
  ignored. Name them `<backend>-<purpose>.sh`, e.g. `jobs/vertex-kcluster.sh`.
  The workspace README records *what* was run and what it produced; the job
  script records *how*, and should stay runnable from a bare clone once the
  data symlinks and config are back in place.
