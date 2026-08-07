import os
from collections import defaultdict

import torch
from google.cloud.aiplatform.prediction.predictor import Predictor


class Phi2Predictor(Predictor):

    def __init__(self):
        super().__init__()
        self.llm = None
        self.pred_params = dict()

    def load(self, artifacts_uri: str) -> None:
        """Load the Phi-2 model and tokenizer."""
        # Imported here so the build host can inspect this class without the
        # full kcluster[local] stack; inside the container both are installed.
        from google.cloud.aiplatform.utils import prediction_utils
        from transformers.utils.import_utils import is_flash_attn_2_available

        from kcluster.engine.local import LargeLangModel

        kwargs = dict(trust_remote_code=True, torch_dtype=torch.float16)
        if is_flash_attn_2_available():
            kwargs["attn_implementation"] = "flash_attention_2"

        # Download model artifacts from the URI if provided
        if artifacts_uri:
            prediction_utils.download_model_artifacts(artifacts_uri)

        model_path = os.path.join(".", "model")
        self.llm = LargeLangModel(model_path, **kwargs)
        self.llm.model.eval()  # Set the model to evaluation mode

    def preprocess(self, prediction_input: dict) -> list[dict]:
        """Preprocesses the input for prediction.
        prediction_input = {"instances": [...], "parameters": {...}}
        """
        self.pred_params = prediction_input.get("parameters", {})
        return prediction_input["instances"]

    def predict(self, instances: list[dict]) -> dict[str, list]:
        """Performs prediction with custom logic.
        instances = [
            {"id": str, "text": str, "context": Optional[str], "purpose": str, "config": Optional[dict]},
            {"id": str, "text": str, "context": Optional[str], "purpose": str, "config": Optional[dict]},
        ]
        """
        # Map instance indices to their purposes
        purpose_mapping = defaultdict(list)
        for idx, item in enumerate(instances):
            purpose_mapping[item["purpose"]].append(idx)

        # Initialize predictions list to be filled later
        predictions = [None] * len(instances)

        # Make predictions for each purpose
        for p in purpose_mapping:
            texts, contexts = [], []
            for idx in purpose_mapping[p]:
                item = instances[idx]
                texts.append(item["text"])
                if "context" in item:
                    contexts.append(item["context"])
                self.pred_params.setdefault(p, {}).update(item.get("config", {}))

            assert (not contexts) or (len(contexts) == len(texts)), "Contexts and texts must have the equal length"

            with torch.inference_mode():
                if contexts:
                    preds = getattr(self.llm, p)(texts, contexts, **(self.pred_params.get(p, {})))
                else:
                    preds = getattr(self.llm, p)(texts, **(self.pred_params.get(p, {})))

            if isinstance(preds, torch.Tensor):
                preds = preds.tolist()

            for idx, pred in zip(purpose_mapping[p], preds, strict=True):
                predictions[idx] = pred

        # postprocessing
        return {"predictions": predictions}
