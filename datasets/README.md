# Dataset workspaces

One directory per dataset KCluster has been applied to. Each workspace holds
the course-specific *driver* scripts — the glue that turns one particular
course export into the Question JSONL format — while the reusable *format
adapters* they call (ScienceQA JSON, OLI HTML, DataShop exports) live in
`kcluster/io/loaders/` and ship with the package. Workspaces are tracked in
the repo but are **never** part of the `kcluster` wheel.

## Onboarding a dataset

1. Create `datasets/<name>/` for the driver scripts.
2. Symlink `datasets/<name>/data` at the dataset directory, which lives outside
   this repository. The name is git-ignored, so the link never gets committed.
3. Write the driver, defaulting its paths to `data/raw/...` and `data/processed/`.

## The three tiers

The dataset directory behind the symlink holds `raw/`, `interim/` and
`processed/`. Two questions place every file, in order:

- **Could I regenerate it from something else in this directory?** If no, it is
  `raw/` — the only tier that is irreplaceable and the only one worth backing
  up. A DataShop export is raw however tidy it looks: you cannot rebuild it,
  only re-download it. Keep the download whole; a column-reduced copy of an
  export is a derived file, not the export.
- **Is it what KCluster runs on?** Only `<ds>.jsonl` is (the question bank), so
  that is `processed/`; everything else you generated is `interim/`, including
  the driver's `<ds>_student-step-minimal.txt` — an untagged file is an input to
  `kcluster tag`, regenerable from the raw export like any other working file.

That second cut is what makes `processed/` the sync unit — the tier that
travels to a cluster. Keeping it to the banks means "what do I need over there?"
is never a judgement call.

Three consequences worth stating:

- **One `<ds>` id threads data through results.** The jsonl stem is the dataset
  id: `elearning22-mcq.jsonl` → `elearning22-mcq_student-step-minimal.txt` →
  `results/<run>/kc/elearning22-mcq_*-kc.csv` →
  `results/<run>/elearning22-mcq_student-step-tagged.txt`. Never rename it
  mid-pipeline.
- **The two student-step files are named for their stage.** `-minimal` has no KC
  model of KCluster's making (expert models may ride along); `-tagged` is what
  `kcluster tag` wrote and what gets scored. The suffixes are `MINIMAL_SUFFIX` /
  `TAGGED_SUFFIX` in `kcluster.io.student_step`, so a driver spells the
  convention once.
- **A tagged file belongs to a run, not to the dataset.** One minimal file is
  common to every run; joining a particular run's KC models onto it produces a
  file only that run explains, so `kcluster tag` writes it into that result dir.
  Datasets clustered in banks (one result dir per course under a work dir) need
  no joint KC table on disk for this: the tagger concatenates the parts of each
  model in memory, namespacing each part's labels by its bank (`EPLA-Physics:
  Newton's third law`) so the banks keep disjoint KC spaces.

One directory per *data source*, not per paper: two DataShop exports of the same
course share the expensive artifact (the course HTML), so they stay together —
or, where they are large enough to warrant their own directory, the second
symlinks the shared export rather than copying it.

## Conventions

- Drivers import from the installed `kcluster` package (`pip install -e .`).
- **Never commit data** (student data, DataShop exports, course-derived question
  banks, images, run outputs); the pre-commit guard enforces this. No export is
  stored in this repository, not even ignored — that is what the `data` symlink
  is for, and every driver also takes its input paths as arguments.
- Scripts that *ran* KCluster on a workspace go in `<name>/jobs/`, git-ignored
  alongside `data`. A driver is reproducible and belongs in the repo; a job is
  one execution on one backend, carrying cluster paths and GCP identifiers that
  are local by nature — the same reason `vertex.toml` is ignored. Name them
  `<backend>-<purpose>.sh`, e.g. `jobs/vertex-kcluster.sh`. The workspace README
  records *what* was run and what it produced; the job script records *how*, and
  should stay runnable from a bare clone once the data symlink and config are
  back in place.
- Exploratory notebooks are git-ignored (`datasets/**/*.ipynb`): their saved
  outputs embed rows of the data. The driver is the record of what they found.
