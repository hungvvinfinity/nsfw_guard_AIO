"""ComfyUI node schema and fail-closed moderation orchestration."""

import json
import math
from numbers import Real

from .classifiers import Backend, DEFAULT_IMAGE_MODEL, DEFAULT_TEXT_MODEL
from .errors import GuardError, NSFWDetectedError


def _model_source(value, modality):
    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_TEXT_MODEL if modality == "prompt" else DEFAULT_IMAGE_MODEL
    if not isinstance(value, str):
        raise GuardError("invalid_model", "The classifier must be a Hugging Face model ID or local directory string.", modality)
    return value.strip()


def _threshold(value, modality):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise GuardError("invalid_sensitivity", "Sensitivity must be a finite number between 0 and 1.", modality)
    return 1.0 - float(value)


def _validate_assessment(assessment, modality):
    score = assessment.confidence
    if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score) or not 0 <= score <= 1:
        raise GuardError("invalid_prediction", "The classifier returned an invalid probability.", modality)


class NSFWGuardAIO:
    CATEGORY = "moderation"
    FUNCTION = "guard"
    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("prompt", "image", "result_json")
    OUTPUT_NODE = True
    DESCRIPTION = "Check prompt/image content before passing it downstream. NSFW content stops the current workflow with a JSON execution error."

    def __init__(self, backend=None):
        self._backend = backend if backend is not None else Backend()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "Higher values block more prompts. NSFW threshold = 1 - sensitivity."}),
                "image_sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "Higher values block more images. NSFW threshold = 1 - sensitivity."}),
            },
            "optional": {
                "model_text_classification": ("STRING", {"default": DEFAULT_TEXT_MODEL}),
                "model_image_classification": ("STRING", {"default": DEFAULT_IMAGE_MODEL}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
            },
        }

    def guard(self, prompt_sensitivity=0.5, image_sensitivity=0.5, model_text_classification=None, model_image_classification=None, prompt=None, image=None):
        prompt_threshold = _threshold(prompt_sensitivity, "prompt")
        image_threshold = _threshold(image_sensitivity, "image")
        if prompt is not None and not isinstance(prompt, str):
            raise GuardError("invalid_prompt", "Prompt must be a string.", "prompt")
        has_prompt = isinstance(prompt, str) and bool(prompt.strip())
        has_image = image is not None
        if not has_prompt and not has_image:
            raise GuardError("no_content", "Supply a nonblank prompt, an image, or both.")

        text_source = _model_source(model_text_classification, "prompt") if has_prompt else None
        image_source = _model_source(model_image_classification, "image") if has_image else None
        if has_image:
            self._backend.validate_image(image)
        if has_prompt:
            assessment = self._backend.classify_prompt(text_source, prompt)
            _validate_assessment(assessment, "prompt")
            if assessment.confidence >= prompt_threshold:
                raise NSFWDetectedError("prompt", float(assessment.confidence))
        if has_image:
            assessment = self._backend.classify_image(image_source, image)
            _validate_assessment(assessment, "image")
            if assessment.confidence >= image_threshold:
                raise NSFWDetectedError("image", float(assessment.confidence), assessment.index)

        safe_json = json.dumps({"status": "safe"})
        out_image = image
        if out_image is None:
            try:
                import execution
                if hasattr(execution, "ExecutionBlocker"):
                    out_image = execution.ExecutionBlocker("NSFW Guard: No image provided to guard node")
            except Exception:
                pass
        return {"ui": {"text": [safe_json]}, "result": (prompt if has_prompt else "", out_image, safe_json)}


NODE_CLASS_MAPPINGS = {"NSFWGuardAIO": NSFWGuardAIO}
NODE_DISPLAY_NAME_MAPPINGS = {"NSFWGuardAIO": "NSFW Guard All in One"}
