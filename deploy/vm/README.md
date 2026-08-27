# L4 VM serving

A stop/start GPU VM running `kcluster serve`, reached through an SSH tunnel.
This is the interactive counterpart of the Vertex batch backend: the same
`LargeLangModel` code behind an HTTP API on `localhost`, so a notebook or a
coding agent on your laptop can call the model programmatically, and a
different model is a restart with a different `--llm_path`. Nothing is
exposed to the internet and the VM costs only its disk while stopped.

Everything here is host-side tooling; none of it ships in the wheel.

## One-time setup

1. **Create the VM** (spot is fine — a preemption is a restart, the disk
   survives). A Deep Learning VM image supplies the NVIDIA driver; its own
   conda environment (Python 3.10, torch 2.9) goes unused — `uv sync` below
   installs the locked Python 3.12 + torch into the repo's venv. Families
   rotate; if the one here is gone, pick a current one with
   `gcloud compute images list --project deeplearning-platform-release --filter="family~pytorch" --format="value(family)"`:

       gcloud compute instances create kcluster-l4 --zone us-central1-a \
           --machine-type g2-standard-4 --accelerator type=nvidia-l4,count=1 \
           --maintenance-policy TERMINATE --provisioning-model SPOT \
           --image-project deeplearning-platform-release \
           --image-family pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
           --boot-disk-size 100GB --boot-disk-type pd-balanced \
           --scopes storage-ro

   Needs Compute Engine GPU quota (`NVIDIA_L4_GPUS` /
   `PREEMPTIBLE_NVIDIA_L4_GPUS` in the region) — separate from the Vertex
   quota the batch jobs use.

   The VM gets an ephemeral external IP so it can reach PyPI, GitHub and
   GCS. The `default` network admits only SSH from the internet and
   `kcluster serve` binds localhost, so nothing else is reachable; SSH can
   still go through IAP (`--tunnel-through-iap`). To run without any public
   address instead, add `--no-address` and give the project outbound access
   first — Cloud NAT (`gcloud compute routers create nat-router --network
   default --region us-central1`, then `gcloud compute routers nats create
   nat-config --router nat-router --region us-central1
   --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges`), a billed
   per-region resource. Either way, enabling Private Google Access on the
   subnet (`gcloud compute networks subnets update default --region
   us-central1 --enable-private-ip-google-access`) is free and routes GCS
   and Artifact Registry traffic internally.

2. **Install kcluster and the model** on the VM:

       gcloud compute ssh kcluster-l4 --zone us-central1-a --tunnel-through-iap
       curl -LsSf https://astral.sh/uv/install.sh | sh
       git clone https://github.com/weiyumou/KCluster ~/kcluster && cd ~/kcluster
       uv sync --extra local --extra serve
       sudo mkdir -p /opt/models && sudo chown $USER /opt/models
       gsutil -m cp -r gs://<bucket>/llm/phi-2 /opt/models/     # or: uv run hf download microsoft/phi-2 --local-dir /opt/models/phi-2

   Try it by hand first:

       uv run kcluster serve --llm_path /opt/models/phi-2 --trust_remote_code
       curl -s localhost:8080/health

3. **Run it as a service** so it comes up on every start, and stop the VM
   when idle:

       sudo cp deploy/vm/systemd/* /etc/systemd/system/
       sudo systemctl daemon-reload
       sudo systemctl enable --now kcluster-serve@$USER kcluster-idle-stop.timer

   Edit the `Environment=` lines in `kcluster-serve@.service` for the model path
   and flags; `kcluster-idle-stop` powers the VM off after
   `IDLE_MINUTES` (default 60) without a request, which GCE records as a
   stopped instance — no credentials on the VM needed.

## Daily loop

    gcloud compute instances start kcluster-l4 --zone us-central1-a
    gcloud compute ssh kcluster-l4 --zone us-central1-a --tunnel-through-iap -- -N -L 8080:localhost:8080 &
    curl -s localhost:8080/health          # ~1–2 min after start: {"status": "ok", "model_id": "phi-2", ...}
    ...
    gcloud compute instances stop kcluster-l4 --zone us-central1-a   # or let the idle timer do it

## Using it

The routes are the engine's methods; `/openapi.json` (or `/docs` in a
browser) has the schemas. `complete_prompts` forwards unknown fields to HF
`generate`, so decoding experiments need no server change:

    curl -s localhost:8080/complete_prompts -H 'Content-Type: application/json' -d '{
      "prompts": ["Remark: This question tests whether the student understands the concept of"],
      "stop_tokens": [".", ","], "max_new_tokens": 20, "num_beams": 5, "length_penalty": -0.1}'
    # {"model_id": "phi-2", "result": [" ..."]}

From Python, `HttpLangModel` mirrors `LargeLangModel`, so the tasks that
only use the engine surface run unmodified:

    from kcluster.engine.http import HttpLangModel
    from kcluster.tasks.concept import extract_concepts
    llm = HttpLangModel("http://localhost:8080")
    extract_concepts(llm, questions, batch_size=16)

Covered: `extract_concepts`, `extract_question_embeds`, `validate_mcq`,
`sort_questions`. Not covered: `generate_mcq`, which hands the tokenizer
object to `generate` — run that in-process on the VM.

## Swapping the model

`kcluster serve --llm_path /opt/models/<other> [--dtype bfloat16] [--quantize 4bit]`
(4-/8-bit needs `pip install bitsandbytes` in the VM's env). Startup checks
that the tokenizer marks context/text pairs with `token_type_ids`, which
`log_prob` and `encode` rely on, and refuses otherwise. To A/B two models,
run a second instance on another port and tunnel both. The prompts and
thresholds in the pipeline are calibrated on Phi-2; a new model needs its
own calibration, which is what the A/B is for. Every response carries
`model_id`, so logged results stay self-describing.
