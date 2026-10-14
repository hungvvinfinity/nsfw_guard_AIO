import importlib.util
import itertools
import json
from pathlib import Path
import unittest

from nsfw_guard.classifiers import Assessment, DEFAULT_IMAGE_MODEL, DEFAULT_TEXT_MODEL
from nsfw_guard.errors import GuardError, NSFWDetectedError
from nsfw_guard.nodes import NSFWGuardAIO


class FakeBackend:
    def __init__(self, prompt_score=0.1, image_score=0.1):
        self.prompt_score = prompt_score
        self.image_score = image_score
        self.calls = []

    def validate_image(self, image):
        self.calls.append(("validate", image))

    def classify_prompt(self, source, prompt):
        self.calls.append(("prompt", source, prompt))
        return Assessment(self.prompt_score)

    def classify_image(self, source, image):
        self.calls.append(("image", source, image))
        return Assessment(self.image_score, 2)


class NodeTests(unittest.TestCase):
    def test_all_content_and_model_presence_combinations(self):
        image = object()
        for has_prompt, has_image, text_model, image_model in itertools.product((False, True), repeat=4):
            with self.subTest(prompt=has_prompt, image=has_image, text_model=text_model, image_model=image_model):
                backend = FakeBackend()
                kwargs = {
                    "prompt": "safe prompt" if has_prompt else None,
                    "image": image if has_image else None,
                    "model_text_classification": DEFAULT_TEXT_MODEL if text_model else None,
                    "model_image_classification": DEFAULT_IMAGE_MODEL if image_model else None,
                }
                error = "no_content" if not (has_prompt or has_image) else None
                if error:
                    with self.assertRaises(GuardError) as caught:
                        NSFWGuardAIO(backend).guard(**kwargs)
                    self.assertEqual(json.loads(str(caught.exception))["error"], error)
                    self.assertEqual(backend.calls, [])
                else:
                    result = NSFWGuardAIO(backend).guard(**kwargs)
                    self.assertEqual(result["result"][0], kwargs["prompt"] or "")
                    if kwargs["image"] is None:
                        self.assertTrue(result["result"][1] is None or getattr(result["result"][1], "__class__", None).__name__ == "ExecutionBlocker")
                    else:
                        self.assertIs(result["result"][1], kwargs["image"])
                    self.assertEqual(json.loads(result["result"][2]), {"status": "safe"})
                    self.assertEqual([x[0] for x in backend.calls if x[0] != "validate"],
                                     (["prompt"] if has_prompt else []) + (["image"] if has_image else []))

    def test_omitted_and_blank_models_use_defaults(self):
        for value in (None, "", " \n"):
            backend = FakeBackend()
            NSFWGuardAIO(backend).guard(prompt="safe", image=object(),
                model_text_classification=value, model_image_classification=value)
            self.assertEqual(backend.calls[1][1], DEFAULT_TEXT_MODEL)
            self.assertEqual(backend.calls[2][1], DEFAULT_IMAGE_MODEL)

    def test_prompt_json_and_short_circuit(self):
        backend = FakeBackend(prompt_score=0.97, image_score=0.99)
        with self.assertRaises(NSFWDetectedError) as caught:
            NSFWGuardAIO(backend).guard(prompt="input", image=object(), model_text_classification="text", model_image_classification="image")
        self.assertEqual(json.loads(str(caught.exception)), {"reason": "Prompt contains NSFW content", "confidence": 0.97, "type": "prompt"})
        self.assertNotIn("image", [x[0] for x in backend.calls])

    def test_image_json_and_batch_index(self):
        with self.assertRaises(NSFWDetectedError) as caught:
            NSFWGuardAIO(FakeBackend(image_score=0.92)).guard(image=object(), model_image_classification="image")
        payload = json.loads(str(caught.exception))
        self.assertEqual(set(payload), {"reason", "confidence", "type"})
        self.assertEqual(payload["type"], "image")
        self.assertEqual(payload["confidence"], 0.92)
        self.assertIn("index 2", payload["reason"])

    def test_sensitivity_direction_and_equality_for_each_modality(self):
        for modality in ("prompt", "image"):
            kwargs = {"prompt": "safe", "model_text_classification": "text"} if modality == "prompt" else {"image": object(), "model_image_classification": "image"}
            for sensitivity, score, blocked in [(0, 0.99, False), (0, 1.0, True), (0.5, 0.49, False), (0.5, 0.5, True), (0.75, 0.3, True), (1, 0.0, True)]:
                with self.subTest(modality=modality, sensitivity=sensitivity, score=score):
                    backend = FakeBackend(prompt_score=score, image_score=score)
                    kwargs[f"{modality}_sensitivity"] = sensitivity
                    if blocked:
                        with self.assertRaises(NSFWDetectedError):
                            NSFWGuardAIO(backend).guard(**kwargs)
                    else:
                        NSFWGuardAIO(backend).guard(**kwargs)

    def test_invalid_sensitivity_and_predictions_fail_closed(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, "0.5", True, None):
            with self.subTest(value=value):
                with self.assertRaises(GuardError) as caught:
                    NSFWGuardAIO(FakeBackend()).guard(prompt="text", model_text_classification="text", prompt_sensitivity=value)
                self.assertEqual(caught.exception.payload["error"], "invalid_sensitivity")
        for score in (float("nan"), float("inf"), -0.1, 1.1, True):
            with self.assertRaises(GuardError) as caught:
                NSFWGuardAIO(FakeBackend(prompt_score=score)).guard(prompt="text", model_text_classification="text")
            self.assertEqual(caught.exception.payload["error"], "invalid_prediction")

    def test_blank_prompt_is_absent_and_unused_models_are_ignored(self):
        backend = FakeBackend()
        output = NSFWGuardAIO(backend).guard(prompt=" \n\t", image=object(), model_text_classification=object(), model_image_classification="image")
        self.assertEqual(output["result"][0], "")
        self.assertNotIn("prompt", [x[0] for x in backend.calls])
        for prompt in (None, "", " \n"):
            with self.assertRaises(GuardError) as caught:
                NSFWGuardAIO(FakeBackend()).guard(prompt=prompt)
            self.assertEqual(caught.exception.payload["error"], "no_content")

    def test_invalid_prompt_and_matching_model(self):
        for prompt in (123, [], {}):
            with self.assertRaises(GuardError) as caught:
                NSFWGuardAIO(FakeBackend()).guard(prompt=prompt)
            self.assertEqual(caught.exception.payload["error"], "invalid_prompt")
        for model in (123, {}):
            with self.assertRaises(GuardError) as caught:
                NSFWGuardAIO(FakeBackend()).guard(prompt="text", model_text_classification=model)
            self.assertEqual(caught.exception.payload["error"], "invalid_model")

    def test_error_propagation_produces_no_safe_outputs(self):
        backend = FakeBackend()
        def fail(*args):
            raise GuardError("inference_failed", "Inference failed", "prompt")
        backend.classify_prompt = fail
        with self.assertRaises(GuardError):
            NSFWGuardAIO(backend).guard(prompt="text", model_text_classification="text")

    def test_source_stripped_but_prompt_unchanged(self):
        backend = FakeBackend()
        prompt = "  safe prompt \n"
        output = NSFWGuardAIO(backend).guard(prompt=prompt, model_text_classification=" text ")
        self.assertEqual(output["result"][0], prompt)
        self.assertEqual(backend.calls, [("prompt", "text", prompt)])

    def test_registration_schema_and_plugin_entrypoint(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("test_comfy_plugin", root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            self.assertEqual(set(module.NODE_CLASS_MAPPINGS), {"NSFWGuardAIO"})
            node = module.NODE_CLASS_MAPPINGS["NSFWGuardAIO"]
            self.assertTrue(node.OUTPUT_NODE)
            self.assertEqual(node.RETURN_TYPES, ("STRING", "IMAGE", "STRING"))
            schema = node.INPUT_TYPES()
            self.assertEqual(set(schema["optional"]), {"model_text_classification", "model_image_classification", "prompt", "image"})
            for slider in schema["required"].values():
                self.assertEqual(slider[1]["display"], "slider")
        finally:
            for key in list(sys.modules):
                if key == spec.name or key.startswith(spec.name + "."):
                    del sys.modules[key]


if __name__ == "__main__":
    unittest.main()
