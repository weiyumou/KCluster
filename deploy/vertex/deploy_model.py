"""Push the built serving image and register the model on Vertex AI.

Prints the uploaded model's resource name — put its model id (and version)
into your vertex TOML config so the ``kcluster vertex-*`` commands can find
it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google.cloud import aiplatform
from google.cloud.aiplatform.prediction import LocalModel
from run_local import LOCAL_MODEL_SPECS


def main(args):
    aiplatform.init(project=args.project_id, location=args.region)

    local_model = LocalModel(serving_container_image_uri=args.image_uri, **LOCAL_MODEL_SPECS)
    print(local_model.get_serving_container_spec())
    local_model.push_image()

    model = aiplatform.Model.upload(
        serving_container_image_uri=args.image_uri,
        artifact_uri=args.artifact_uri,
        display_name=args.display_name,
        description=args.description,
        **LOCAL_MODEL_SPECS,
    )
    print(f"Uploaded model: {model.resource_name}@{model.version_id}")
    print("Set 'model_id' (and 'model_version') in your vertex TOML config accordingly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy the KCluster serving image to Vertex AI")
    parser.add_argument("--project_id", required=True, type=str, help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", type=str, help="Deployment region")
    parser.add_argument("--image_uri", required=True, type=str, help="The URI of the container image to deploy")
    parser.add_argument("--artifact_uri", required=True, type=str,
                        help="GCS directory with the model artifacts, e.g. gs://<bucket>/llm/phi-2")
    parser.add_argument("--display_name", default="kcluster-phi2", type=str, help="Model display name")
    parser.add_argument("--description", default="KCluster with a Phi-2 backend", type=str,
                        help="Model description")
    cl_args = parser.parse_args()
    main(cl_args)
