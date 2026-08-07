import logging
import traceback

try:
    from fastapi import HTTPException, Request, Response
except ImportError:
    raise ImportError(
        "FastAPI is not installed and is required to build model servers. "
        'Please install the SDK using `pip install "google-cloud-aiplatform[prediction]>=1.16.0"`.'
    )

from google.cloud.aiplatform.prediction import DefaultSerializer, PredictionHandler, handler_utils


class Phi2PredictionHandler(PredictionHandler):
    """Useful for local debugging and testing. Keep the implementation same as the default"""

    async def handle(self, request: Request) -> Response:
        request_body = await request.body()

        # logging.info(f"Received request: {request_body}")

        content_type = handler_utils.get_content_type_from_headers(request.headers)
        prediction_input = DefaultSerializer.deserialize(request_body, content_type)

        # logging.info(f"Received prediction input: {prediction_input}")

        try:
            prediction_results = self._predictor.postprocess(
                self._predictor.predict(self._predictor.preprocess(prediction_input))
            )
        except HTTPException:
            raise
        except Exception as exception:
            error_message = (
                "The following exception has occurred: {}. Arguments: {}.".format(
                    type(exception).__name__, exception.args
                )
            )
            logging.info("{}\\nTraceback: {}".format(error_message, traceback.format_exc()))

            # Converts all other exceptions to HTTPException.
            raise HTTPException(status_code=500, detail=error_message)

        accept = handler_utils.get_accept_from_headers(request.headers)
        data = DefaultSerializer.serialize(prediction_results, accept)
        return Response(content=data, media_type=accept)
