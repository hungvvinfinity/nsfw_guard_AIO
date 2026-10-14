# NSFW Guard All in One for ComfyUI

A custom node that checks prompts and image batches before passing them downstream. An NSFW detection raises a JSON error and stops the current ComfyUI workflow.

## Install

1. Put this directory in `ComfyUI/custom_nodes/nsfw_guard_AIO`.
2. Install dependencies using **the Python environment that runs ComfyUI**:

   ```bash
   python -m pip install -r custom_nodes/nsfw_guard_AIO/requirements.txt
   ```

   For Windows portable ComfyUI, run this from its top-level directory:

   ```powershell
   .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\nsfw_guard_AIO\requirements.txt
   ```

3. Restart ComfyUI. Add **moderation → NSFW Guard All in One**.

Python 3.10+ and PyTorch 2.5+ are required. The node targets Transformers 5.19+, huggingface-hub 2.2+, and tokenizers 0.23.3+ (below 0.24). To upgrade an existing installation, run `python -m pip install --upgrade -r custom_nodes/nsfw_guard_AIO/requirements.txt` in ComfyUI's Python environment, then restart ComfyUI. ComfyUI's existing PyTorch installation is reused when it satisfies the requirements.

## Inputs and outputs

| Input | Default | Behavior |
| --- | --- | --- |
| `model_text_classification` | `scaleTech/myplaygirl-nsfw-classifier` | Hugging Face ID or local text-model directory |
| `model_image_classification` | `Falconsai/nsfw_image_detection` | Hugging Face ID or local image-model directory |
| `prompt` | Empty | Optional multiline text; convert the widget to an input to connect another node |
| `image` | Unconnected | Optional ComfyUI image batch, `[batch, height, width, channels]` |
| `prompt_sensitivity` | `0.50` | Slider, `0–1`; higher blocks more prompts |
| `image_sensitivity` | `0.50` | Slider, `0–1`; higher blocks more images |

The four main inputs are accompanied by two sensitivity controls. The probability threshold is `1 - sensitivity`, and an NSFW score **equal to or above** that threshold stops execution. At sensitivity `1`, every supplied item is blocked; at `0`, only an NSFW score of exactly `1` is blocked. These scores are model probabilities, not calibrated guarantees.

Outputs are `prompt`, `image`, and `result_json`. On success, supplied content passes through unchanged and `result_json` is `{"status": "safe"}`. An absent prompt returns `""`; an absent image returns `None`. Connect only the modality outputs you supplied. The success JSON is also available in the node's ComfyUI history output under `text`.

| Supplied content | Required classifier | Result |
| --- | --- | --- |
| Prompt only | Text only | Check prompt; image model is unused |
| Image only | Image only | Check every image; text model is unused |
| Both | Both | Check prompt first, then all images if prompt passes |
| Neither / whitespace-only prompt | Neither | Stop with `no_content` |
| Content without a model setting | Built-in default classifier | Download automatically if absent from disk cache |

Omitted or blank model fields use the built-in defaults in both UI and API workflows. No separate loader nodes are needed. Missing model files download automatically on first use. Invalid unused model settings are ignored. Download, authentication, input-validation, and inference failures stop execution rather than allowing unchecked content through.

## Wire the guard into the workflow

For text generation, route `guard.prompt → CLIPTextEncode.text`. For image processing, route `LoadImage → guard.image → downstream image node`. All operations that must wait for moderation need a dependency on the guard's checked outputs. Do not also route the original content around the guard.

The node runs even as a standalone output node. When detection raises an error, ComfyUI stops the current workflow; other queued workflows are unaffected. Operations completed before the guard ran cannot be undone, and independent branches are not guaranteed to wait for it.

Drag-and-drop UI workflows and ready-to-submit API workflows are in [examples](examples/README.md): prompt-only, image-only, and combined. Open the `*_workflow.json` files in the graphical editor; the other JSON files use ComfyUI's API format.

## Detection JSON

Prompt detection:

```json
{"reason": "Prompt contains NSFW content", "confidence": 0.97, "type": "prompt"}
```

Image detection (indices start at zero):

```json
{"reason": "Image at batch index 2 contains NSFW content", "confidence": 0.92, "type": "image"}
```

ComfyUI sends this JSON string in `execution_error.data.exception_message` over its WebSocket connection and stores the same event in history status messages. The exception type ends in `NSFWDetectedError`. Parse the trimmed message:

```python
payload = json.loads(event["data"]["exception_message"].strip())
```

The initial HTTP `/prompt` response only acknowledges queue submission; moderation happens during execution. NSFW detection does not return normal node outputs. If both modalities would be flagged, the prompt error takes precedence and image inference is skipped.

Other failures use a separate schema without a fabricated confidence:

```json
{"error": "model_load_failed", "reason": "Could not load the classifier.", "type": "image"}
```

Error codes include `no_content`, `invalid_model`, `invalid_prompt`, `invalid_image`, `invalid_sensitivity`, `dependency_missing`, `model_access_denied`, `model_load_failed`, `unsupported_model`, `ambiguous_labels`, `invalid_prediction`, and `inference_failed`. `no_content` has no `type`.

## Models, authentication, and caching

- Text: [scaleTech/myplaygirl-nsfw-classifier](https://huggingface.co/scaleTech/myplaygirl-nsfw-classifier), with documented class `0 = NSFW`, `1 = SFW`. It is an English text classifier. Long prompts use overlapping windows with up to 64 tokens of overlap; the maximum NSFW score determines the result.
- Image: [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection). This public, ungated model downloads without authentication. Custom gated models still require accepting their repository conditions and authenticating with `HF_TOKEN` or `hf auth login`. The node never substitutes a different model.
- Models download directly into `ComfyUI/models/nsfw_guard/<owner>/<repository>/`, using ComfyUI's configured `folder_paths.models_dir`. The Hugging Face CLI model cache and `HF_HOME` are not used for model storage. For example, the defaults live in `models/nsfw_guard/scaleTech/myplaygirl-nsfw-classifier/` and `models/nsfw_guard/Falconsai/nsfw_image_detection/`. Transformers loads only these local files. Completed downloads are reused without Hub checks; delete `.download_complete` in the model directory to retry or refresh a download. `HF_HUB_OFFLINE=1` requires an already completed download or an explicit local directory. Outside ComfyUI, the fallback is this custom node's `models/nsfw_guard/` directory. Hugging Face may keep download metadata in `.cache/huggingface/` inside each model directory; weights remain in that directory. When required for custom models, authentication uses `HF_TOKEN` or your authorized login. A local directory must contain complete model and tokenizer/processor files, including safetensors weights.
- CUDA is selected when available; otherwise CPU is used. CUDA errors stop the workflow rather than silently moving models to CPU. Weights use their standard full precision.
- Loading is lazy. A prompt-only run never loads an image classifier. Models and processors are cached by modality, model source, and device in one Python process. Two copies of the node with the same configuration and different image inputs receive the exact same model and ModelPatcher objects; each configuration initializes once. A lock serializes loading/inference. Local directory aliases resolve to the same cache key. Separate processes have separate caches.
- In ComfyUI, each cached classifier owns one `ModelPatcher`; inference calls `comfy.model_management.load_models_gpu` on that same patcher. ComfyUI can offload weights to CPU and reload them into VRAM without reconstructing the classifier. Transformers layers require full loading. Outside ComfyUI, inference uses ordinary PyTorch device placement. Configurations remain cached in CPU memory until process exit; selecting many different models increases RAM use.
- ComfyUI may reuse safe node results when inputs are unchanged. Restart ComfyUI after replacing files inside a local model directory or to release all cached classifier weights.

Alternative classifiers must be binary, mutually exclusive classifiers with descriptive `id2label` values: `NSFW` and `SFW`, `normal`, or `safe` (case-insensitive). Models declaring regression or multilabel tasks are rejected. Only the exact default text model ID receives its documented fallback mapping. All local models, including local copies of that default, must declare descriptive labels in `config.json`; generic `LABEL_0` / `LABEL_1` labels are rejected because their class order cannot be established. For a verified copy of the default scaleTech model, add `"id2label": {"0": "NSFW", "1": "SFW"}` and `"label2id": {"NSFW": 0, "SFW": 1}`. Use the trained class order for other models. Arbitrary remote model code and pickle weights are not loaded.

Images must be nonempty floating-point tensors with finite pixels in `[0, 1]`. One-, three-, and four-channel images are supported and converted to RGB for inference. RGBA conversion drops alpha and checks its RGB content. Every item is checked; the maximum score and its batch index are reported. The original tensor is unchanged.

## Verify

Install test dependencies and run the offline tests:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q
```

Run real-model smoke checks (downloads weights on first use; image requires access):

```bash
python -m nsfw_guard.smoke --modality prompt
python -m nsfw_guard.smoke --modality image
```

The smoke command reports model scores without asserting that a synthetic input must receive a particular label. To check execution errors against a running ComfyUI with this node installed:

```bash
COMFYUI_URL=http://127.0.0.1:8188 python -m pytest -q tests/test_comfyui_integration.py
```

The live check queues a prompt-only workflow at maximum sensitivity and verifies that its dependent node never executes. Without `COMFYUI_URL`, it is skipped.

You can also exercise ComfyUI's actual node loader, validator, and workflow executor without opening a server socket. Supply a local ComfyUI checkout with its dependencies installed in the same Python environment:

```bash
COMFYUI_PATH=/path/to/ComfyUI COMFYUI_GUARD_TEXT_MODEL=/path/to/local/text-model python -m pytest -q tests/test_comfy_executor.py
```

This checks safe image saving, blocked image execution, invalid-model errors, and prompt blocking. Image checks use a small generated local ViT checkpoint. The prompt check uses the supplied text model and is skipped if `COMFYUI_GUARD_TEXT_MODEL` is absent.
