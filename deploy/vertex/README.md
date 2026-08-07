# Vertex AI deployment

Everything needed to run KCluster's model backend in **your own** GCP
project (decision D4: bring-your-own-project — no shared model, bucket, or
image; Vertex Model Registry entries are project-scoped and cannot be
shared). The scripts here are host-side tooling and are not part of the
`kcluster` wheel; run them from the repo's dev environment.

## One-time setup

1. **Model artifacts** — download [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
   (MIT license) and upload it to your bucket:

       hf download microsoft/phi-2 --local-dir phi-2
       gsutil -m cp -r phi-2 gs://<bucket>/llm/phi-2

2. **Build the serving image** (requires Docker; the wheel step goes away
   once kcluster is on PyPI):

       uv build
       python deploy/vertex/build_image.py \
           --project_id <project> --repository <artifact-repo> --image_name kcluster-phi2 \
           --kcluster_package dist/kcluster-<version>-py3-none-any.whl

3. *(Optional)* **Smoke-test locally** — serves the container and sends a
   small mixed-purpose request:

       python deploy/vertex/run_local.py --project_id <project> \
           --image_uri <image-uri> --artifacts_dir <local-phi-2-dir>

4. **Push and register the model**:

       python deploy/vertex/deploy_model.py --project_id <project> \
           --image_uri <image-uri> --artifact_uri gs://<bucket>/llm/phi-2

   Copy the printed model id/version into your vertex TOML config; the
   `kcluster vertex-launch` / `vertex-retrieve` / `vertex-build-kc`
   commands take it from there.

## Layout

- `serving/` — the container source (`src_dir`): `predictor.py` dispatches
  batched instances by `purpose` to the corresponding `LargeLangModel`
  method (`complete_prompts`, `log_prob`, ...) — this is the de-facto RPC
  contract with `kcluster/engine/vertex.py`; `handler.py` is a debug-friendly
  request handler; `requirements.txt` is installed into the image on top of
  the PyTorch base image. The predictor imports `kcluster.engine.local` —
  the model code is single-sourced from the package, not copied.
- `custom.py` — a vendored variant of the aiplatform SDK's CPR image build
  that avoids requiring root inside the container (writable `HOME=/home`,
  `WORKDIR=/usr/app`). It reaches into private SDK internals, which is why
  the `vertex` extra pins `google-cloud-aiplatform<2`.
- `build_image.py` / `run_local.py` / `deploy_model.py` — build, local
  smoke test, and registration, all parameterized by project/region/bucket.
