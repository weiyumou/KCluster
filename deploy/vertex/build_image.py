"""Build the KCluster serving container image for Vertex AI.

The image is built from ``serving/`` (predictor + handler + requirements)
and pushed to the *user's* Artifact Registry — no project identifiers are
baked in. Until kcluster is released on PyPI, pass a locally built wheel
via ``--kcluster_package`` (it is copied into the build context and
installed into the image).
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for custom.py
SERVING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serving")
sys.path.insert(0, SERVING_DIR)  # import predictor/handler as in the container

from custom import MyLocalModel
from google.cloud import aiplatform
from handler import Phi2PredictionHandler
from predictor import Phi2Predictor

BASE_IMAGE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"


def main(args):
    aiplatform.init(project=args.project_id, location=args.region)

    extra_packages = None
    if getattr(args, "kcluster_package", None):
        # The build context is src_dir, so the wheel must live inside it
        wheel_path = shutil.copy(args.kcluster_package, SERVING_DIR)
        extra_packages = [wheel_path]

    local_model = MyLocalModel.build_cpr_model(
        src_dir=SERVING_DIR,
        output_image_uri=f"{args.region}-docker.pkg.dev/{args.project_id}/{args.repository}/{args.image_name}",
        predictor=Phi2Predictor,
        handler=Phi2PredictionHandler,
        requirements_path=os.path.join(SERVING_DIR, "requirements.txt"),
        extra_packages=extra_packages,
        base_image=BASE_IMAGE,
        platform="linux/amd64",
    )

    print(local_model.get_serving_container_spec())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the KCluster serving image for Vertex AI")
    parser.add_argument("--project_id", required=True, type=str, help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", type=str, help="Deployment region")
    parser.add_argument("--repository", required=True, type=str, help="Artifact repository name in GCP")
    parser.add_argument("--image_name", required=True, type=str, help="The name of the Docker image to build")
    parser.add_argument("--kcluster_package", default=argparse.SUPPRESS, type=str,
                        help="Path to a locally built kcluster wheel to install into the image")

    cl_args = parser.parse_args()
    main(cl_args)
