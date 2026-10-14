"""Lazy Transformers inference and a process-wide classifier cache."""

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
from threading import RLock
from typing import Any

from .errors import GuardError

DEFAULT_TEXT_MODEL = "scaleTech/myplaygirl-nsfw-classifier"
DEFAULT_IMAGE_MODEL = "Falconsai/nsfw_image_detection"


@dataclass(frozen=True)
class Assessment:
    confidence: float
    index: int | None = None


@dataclass
class Classifier:
    model: Any
    processor: Any
    nsfw_index: int
    device: str
    patcher: Any = None


def _torch():
    try:
        import torch
    except ImportError:
        raise GuardError("dependency_missing", "PyTorch is required; install the node dependencies in ComfyUI's Python environment.") from None
    return torch


def select_device() -> str:
    try:
        from comfy.model_management import get_torch_device
        return str(get_torch_device())
    except Exception:
        pass
    torch = _torch()
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _check_interrupted():
    # Respect ComfyUI cancellation between inference windows/batch items.
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def _is_oom(exc: Exception) -> bool:
    try:
        from comfy.model_management import is_oom
        if is_oom(exc):
            return True
    except Exception:
        pass
    torch = _torch()
    oom_types = tuple(
        t for t in (getattr(torch.cuda, "OutOfMemoryError", None), getattr(torch, "OutOfMemoryError", None))
        if isinstance(t, type)
    )
    if oom_types and isinstance(exc, oom_types):
        return True
    return "out of memory" in str(exc).lower()


def _preserve_interrupt(exc: Exception):
    if exc.__class__.__name__ == "InterruptProcessingException":
        raise exc
    if _is_oom(exc):
        try:
            from comfy.model_management import soft_empty_cache
            soft_empty_cache()
        except Exception:
            pass
        raise exc


def resolve_nsfw_index(config, modality: str, source: str) -> int:
    if config.num_labels != 2 or getattr(config, "problem_type", None) not in (None, "single_label_classification"):
        raise GuardError("unsupported_model", "The classifier must use two mutually exclusive classes.", modality)
    labels = {int(k): str(v).strip().lower() for k, v in (config.id2label or {}).items()}
    if set(labels) != {0, 1}:
        raise GuardError("ambiguous_labels", "Expected classifier labels for class indices 0 and 1.", modality)
    nsfw = [index for index, label in labels.items() if label == "nsfw"]
    safe = {"sfw", "normal", "safe"}
    if len(nsfw) == 1 and labels[1 - nsfw[0]] in safe:
        return nsfw[0]
    generic = labels == {0: "label_0", 1: "label_1"}
    # Only the exact known model ID has a documented 0=NSFW / 1=SFW layout.
    # A local directory cannot establish model identity or class order.
    if modality == "prompt" and generic and source == DEFAULT_TEXT_MODEL and not Path(source).is_dir():
        return 0
    raise GuardError("ambiguous_labels", "Set id2label to NSFW and SFW (or normal/safe); the NSFW class cannot be determined.", modality)


def _access_denied(exc: Exception) -> bool:
    # Transformers wraps Hugging Face access errors in OSError. Inspect the
    # exception chain without including credential-bearing messages in output.
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        response = getattr(exc, "response", None)
        if exc.__class__.__name__ == "GatedRepoError" or getattr(response, "status_code", None) in (401, 403):
            return True
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return False


def _validate_model_head(model, loading_info, modality: str):
    if not isinstance(loading_info, dict):
        return
    missing_keys = loading_info.get("missing_keys", [])
    if not missing_keys:
        return
    head_indicators = ("classifier", "score", "logits_proj", "heads", "head", "fc", "pre_classifier")
    missing_head_keys = [k for k in missing_keys if any(h in k.lower() for h in head_indicators)]
    if missing_head_keys:
        raise GuardError(
            "invalid_model",
            f"Model checkpoint lacks trained classification head weights: {missing_head_keys}",
            modality,
        )


def model_directory() -> Path:
    try:
        import folder_paths
    except ImportError:
        return Path(__file__).resolve().parent.parent / "models" / "nsfw_guard"
    return Path(folder_paths.models_dir) / "nsfw_guard"


def local_model_source(source: str) -> str:
    path = Path(source).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(source)
    target = model_directory().joinpath(*source.split("/"))
    complete = target / ".download_complete"
    if not complete.is_file():
        if os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}:
            raise OSError("Model is not downloaded in the ComfyUI models directory.")
        snapshot_download(
            repo_id=source,
            local_dir=str(target),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.spm", "*.tiktoken"],
        )
        complete.touch()
    return str(target)


def _allow_patcher_device_assignment(model):
    """Accept ComfyUI's device bookkeeping while retaining the HF getter."""
    model_type = type(model)
    device_property = getattr(model_type, "device", None)
    if not isinstance(device_property, property) or device_property.fset is not None:
        return

    def set_device(self, value):
        # ModelPatcher moves the tensors itself before assigning this metadata.
        # Keep reads derived from the tensors, including after a direct .to().
        pass

    # Change only this instance; do not modify Transformers classes globally.
    model.__class__ = type(
        f"Comfy{model_type.__name__}",
        (model_type,),
        {"__module__": __name__, "device": device_property.setter(set_device)},
    )


def load_classifier(modality: str, source: str, device: str) -> Classifier:
    try:
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError:
        raise GuardError("dependency_missing", "Install requirements.txt in ComfyUI's Python environment.", modality) from None
    try:
        local_source = local_model_source(source)
        common = {"trust_remote_code": False, "local_files_only": True}
        if modality == "prompt":
            res = AutoModelForSequenceClassification.from_pretrained(local_source, use_safetensors=True, output_loading_info=True, **common)
            model, loading_info = res if isinstance(res, tuple) else (res, {})
            _validate_model_head(model, loading_info, modality)
            # The Mistral heuristic also catches this 128k
            # DeBERTa Unigram tokenizer. Preserve its trained pre-tokenizer.
            tokenizer_options = {"fix_mistral_regex": False} if getattr(model.config, "model_type", None) == "deberta-v2" else {}
            processor = AutoTokenizer.from_pretrained(local_source, **tokenizer_options, **common)
            if not processor.is_fast:
                raise GuardError("unsupported_model", "A fast tokenizer is required to check the complete prompt in overlapping windows.", modality)
        else:
            processor = AutoImageProcessor.from_pretrained(local_source, backend="pil", **common)
            res = AutoModelForImageClassification.from_pretrained(local_source, use_safetensors=True, output_loading_info=True, **common)
            model, loading_info = res if isinstance(res, tuple) else (res, {})
            _validate_model_head(model, loading_info, modality)
        nsfw_index = resolve_nsfw_index(model.config, modality, source)
        model.eval()
        try:
            from comfy.model_patcher import ModelPatcher
        except ImportError:
            model.to(device)
            return Classifier(model, processor, nsfw_index, device)
        # Keep one patcher per cached configuration. ComfyUI owns GPU residency;
        # offloading must not discard the Python model or re-read its weights.
        offload_device = _torch().device("cpu")
        model.to(offload_device)
        _allow_patcher_device_assignment(model)
        patcher = ModelPatcher(model, load_device=_torch().device(device), offload_device=offload_device)
        return Classifier(model, processor, nsfw_index, device, patcher)
    except GuardError:
        raise
    except Exception as exc:
        _preserve_interrupt(exc)
        if _access_denied(exc):
            raise GuardError("model_access_denied", "Accept the model's Hugging Face access conditions and set HF_TOKEN or log in with an authorized read token.", modality) from None
        # Do not echo exceptions that could contain credentials or request URLs.
        raise GuardError("model_load_failed", "Could not load the classifier. Check the model ID/local directory, files in ComfyUI/models/nsfw_guard, network access, dependencies, and available memory.", modality) from None


class ModelCache:
    """Share exact model/patcher identities by configuration across all nodes.

    The lock covers initialization and inference, including residency changes.
    Strong references retain CPU models when ComfyUI releases their VRAM.
    """

    def __init__(self, loader=load_classifier, device_selector=select_device):
        self._loader = loader
        self._device_selector = device_selector
        self._entries = {}
        self._lock = RLock()

    @contextmanager
    def use(self, modality: str, source: str):
        with self._lock:
            device = self._device_selector()
            path = Path(source).expanduser()
            source = str(path.resolve()) if path.is_dir() else source.strip()
            key = (modality, source, device)
            if key not in self._entries:
                self._entries[key] = self._loader(modality, source, device)
            classifier = self._entries[key]
            patcher = getattr(classifier, "patcher", None)
            if patcher is not None:
                from comfy.model_management import load_models_gpu
                # Transformers layers do not implement ComfyUI's low-VRAM casts.
                load_models_gpu([patcher], force_full_load=True)
            yield classifier


_MODEL_CACHE = ModelCache()


def _score(logits, nsfw_index: int, modality: str) -> float:
    torch = _torch()
    if logits.ndim != 2 or tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all().item():
        raise GuardError("invalid_prediction", "The classifier returned invalid logits.", modality)
    score = float(torch.softmax(logits.float(), dim=-1)[0, nsfw_index].item())
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise GuardError("invalid_prediction", "The classifier returned an invalid probability.", modality)
    return score


def _context_length(model, tokenizer) -> int:
    candidates = []
    for value in (getattr(model.config, "max_position_embeddings", None), getattr(tokenizer, "model_max_length", None)):
        if isinstance(value, int) and 0 < value < 1_000_000:
            candidates.append(value)
    return min(candidates) if candidates else 512


class Backend:
    def __init__(self, cache=None):
        self.cache = cache if cache is not None else _MODEL_CACHE

    def validate_image(self, image):
        torch = _torch()
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            raise GuardError("invalid_image", "Expected a BHWC image tensor with 1, 3, or 4 channels.", "image")
        if any(size == 0 for size in image.shape) or not image.is_floating_point():
            raise GuardError("invalid_image", "The image batch must be nonempty and use floating-point pixels.", "image")
        min_val, max_val = torch.aminmax(image)
        # Single device reduction and scalar transfer, allowing small epsilon tolerance (1e-3) for fp16/bf16 VAE decoding jitter
        valid = (torch.isfinite(min_val) & torch.isfinite(max_val) & (min_val >= -1e-3) & (max_val <= 1.0 + 1e-3)).item()
        if not valid:
            raise GuardError("invalid_image", "Image pixels must be finite values between 0 and 1.", "image")

    def classify_prompt(self, source: str, prompt: str) -> Assessment:
        torch = _torch()
        with self.cache.use("prompt", source) as classifier:
            try:
                tokenizer = classifier.processor
                max_length = _context_length(classifier.model, tokenizer)
                usable = max_length - tokenizer.num_special_tokens_to_add(pair=False)
                if usable <= 0:
                    raise GuardError("unsupported_model", "The tokenizer has no usable context window.", "prompt")
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    stride=min(64, usable - 1),
                    return_overflowing_tokens=True,
                )
                if encoded["input_ids"].shape[0] == 0:
                    raise GuardError("invalid_prediction", "The tokenizer returned no prompt windows.", "prompt")
                maximum = 0.0
                with torch.inference_mode():
                    for index in range(encoded["input_ids"].shape[0]):
                        _check_interrupted()
                        inputs = {name: encoded[name][index:index + 1].to(classifier.device) for name in tokenizer.model_input_names if name in encoded}
                        score = _score(classifier.model(**inputs).logits, classifier.nsfw_index, "prompt")
                        maximum = max(maximum, score)
                return Assessment(maximum)
            except GuardError:
                raise
            except Exception as exc:
                _preserve_interrupt(exc)
                raise GuardError("inference_failed", "Prompt classification failed; check tokenizer/model compatibility and available memory.", "prompt") from None

    def classify_image(self, source: str, image) -> Assessment:
        torch = _torch()
        with self.cache.use("image", source) as classifier:
            try:
                from PIL import Image

                maximum = -1.0
                flagged_index = 0
                with torch.inference_mode():
                    for index, tensor in enumerate(image):
                        _check_interrupted()
                        pixels = tensor.detach().clamp(0.0, 1.0).mul(255).round().to(device="cpu", dtype=torch.uint8).numpy()
                        if pixels.shape[-1] == 1:
                            pixels = pixels[..., 0]
                        pil_image = Image.fromarray(pixels).convert("RGB")
                        inputs = classifier.processor(images=pil_image, return_tensors="pt")
                        inputs = {name: value.to(classifier.device) for name, value in inputs.items()}
                        score = _score(classifier.model(**inputs).logits, classifier.nsfw_index, "image")
                        if score > maximum:
                            maximum, flagged_index = score, index
                return Assessment(maximum, flagged_index)
            except GuardError:
                raise
            except Exception as exc:
                _preserve_interrupt(exc)
                raise GuardError("inference_failed", "Image classification failed; check processor/model compatibility and available memory.", "image") from None
