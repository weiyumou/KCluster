"""Serve the built container locally and send it a smoke-test request."""

import argparse

from google.cloud import aiplatform
from google.cloud.aiplatform.prediction import LocalModel

LOCAL_MODEL_SPECS = dict(
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
    serving_container_ports=[8080],
    serving_container_environment_variables={
        "VERTEX_CPR_WEB_CONCURRENCY": "1",  # Set the number of workers the server starts
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.6,max_split_size_mb:128"
    }
)


def main(args):
    aiplatform.init(project=args.project_id, location=args.region)

    local_model = LocalModel(serving_container_image_uri=args.image_uri, **LOCAL_MODEL_SPECS)

    print(local_model.get_serving_container_spec())

    with local_model.deploy_to_local_endpoint(artifact_uri=args.artifacts_dir) as local_endpoint:
        local_endpoint.serve()
        print(
            ("Local endpoint is running. "
             f"You can now send requests to it at http://localhost:{local_endpoint.assigned_host_port}/predict")
        )

        import json
        params = {}
        predict_response = local_endpoint.predict(
            request=json.dumps({"instances": [{"text": "Today is", "purpose": "complete_prompts",
                                               "config": {"max_new_tokens": 50, "num_beams": 2, "length_penalty": -0.1,
                                                          "do_sample": False, "stop_tokens": [".", ","]}},
                                              {"text": "Monday", "context": "Today is", "purpose": "log_prob"},
                                              {"text": "sunny", "context": "Today is", "purpose": "log_prob"},
                                              {"text": "I am", "purpose": "complete_prompts"}],
                                "parameters": params}),
            headers={"Content-Type": "application/json"},
        )

        health_check_response = local_endpoint.run_health_check()
        print(predict_response.content)
        print(health_check_response.content)

        import threading
        event = threading.Event()
        event.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the KCluster container locally")
    parser.add_argument("--project_id", required=True, type=str, help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", type=str, help="Deployment region")
    parser.add_argument("--image_uri", required=True, type=str, help="The URI of the container image to serve")
    parser.add_argument("--artifacts_dir", default=None, type=str,
                        help="Local directory with the model artifacts (e.g. a downloaded phi-2 checkout)")
    cl_args = parser.parse_args()
    main(cl_args)
