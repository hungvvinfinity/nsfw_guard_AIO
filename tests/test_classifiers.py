from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from nsfw_guard.classifiers import (
    Assessment, Backend, Classifier, DEFAULT_TEXT_MODEL, ModelCache,
    _allow_patcher_device_assignment, _context_length, _score, load_classifier, resolve_nsfw_index, select_device,
)
from nsfw_guard.errors import GuardError


class PatcherDeviceTests(unittest.TestCase):
    def test_transformers_device_assignment_preserves_tensor_device_and_inference(self):
        from transformers import (
            DebertaV2Config, DebertaV2ForSequenceClassification,
            ViTConfig, ViTForImageClassification,
        )

        models = [
            (DebertaV2ForSequenceClassification(DebertaV2Config(
                vocab_size=16, hidden_size=8, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=16,
            )), {"input_ids": torch.ones(1, 4, dtype=torch.long)}),
            (ViTForImageClassification(ViTConfig(
                image_size=16, patch_size=8, hidden_size=8,
                num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
            )), {"pixel_values": torch.zeros(1, 3, 16, 16)}),
        ]
        for model, inputs in models:
            with self.subTest(model=type(model).__name__):
                original_type = type(model)
                original_property = original_type.device
                model.eval()
                with torch.inference_mode():
                    expected = model(**inputs).logits
                _allow_patcher_device_assignment(model)
                adapted_type = type(model)
                _allow_patcher_device_assignment(model)
                self.assertIs(type(model), adapted_type)
                self.assertIsInstance(model, original_type)
                self.assertIs(original_type.device, original_property)
                self.assertIsNone(original_type.device.fset)
                # Emulate ModelPatcher load and detach bookkeeping.
                for target in ("cuda:0", "cpu"):
                    model.device = torch.device(target)
                    self.assertEqual(model.device, next(model.parameters()).device)
                with torch.inference_mode():
                    torch.testing.assert_close(model(**inputs).logits, expected)
                model.to("meta")
                model.device = torch.device("cpu")
                self.assertEqual(model.device, torch.device("meta"))


class CacheTests(unittest.TestCase):
    def test_reuse_across_instances_and_threads(self):
        loader = Mock(side_effect=lambda *args: object())
        cache = ModelCache(loader, lambda: "cpu")
        first, second = Backend(cache), Backend(cache)
        def use(backend):
            with backend.cache.use("prompt", "text") as model:
                return model
        with ThreadPoolExecutor(max_workers=8) as pool:
            objects = list(pool.map(use, [first, second] * 20))
        self.assertTrue(all(model is objects[0] for model in objects))
        loader.assert_called_once_with("prompt", "text", "cpu")

    def test_modalities_source_and_device_configurations(self):
        loader = Mock(side_effect=lambda *args: object())
        device = Mock(return_value="cpu")
        cache = ModelCache(loader, device)
        for modality, source in [("prompt", "a"), ("image", "b"), ("prompt", "a"), ("prompt", "c"), ("image", "b")]:
            with cache.use(modality, source):
                pass
        self.assertEqual(loader.call_count, 3)
        self.assertEqual(len(cache._entries), 3)
        device.return_value = "cuda:0"
        with cache.use("prompt", "c"):
            pass
        self.assertEqual(loader.call_count, 4)
        self.assertEqual(len(cache._entries), 4)
        with cache.use("prompt", "a"):
            pass
        self.assertEqual(loader.call_count, 5)
        device.return_value = "cpu"
        with cache.use("prompt", "a"):
            pass
        self.assertEqual(loader.call_count, 5)

    def test_two_nodes_different_images_share_model_and_patcher(self):
        from nsfw_guard.nodes import NSFWGuardAIO
        model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[10.0, -10.0]])))
        processor = Mock(return_value={"pixel_values": torch.zeros(1, 3, 2, 2)})
        patcher = object()
        classifier = Classifier(model, processor, 1, "cpu", patcher)
        loader = Mock(return_value=classifier)
        cache = ModelCache(loader, lambda: "cpu")
        manager = SimpleNamespace(load_models_gpu=Mock())
        with patch.dict("sys.modules", {"comfy.model_management": manager}):
            for image in (torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 2, 3)):
                NSFWGuardAIO(Backend(cache)).guard(image=image)
        loader.assert_called_once()
        self.assertEqual(model.call_count, 2)
        self.assertEqual(manager.load_models_gpu.call_count, 2)
        for call in manager.load_models_gpu.call_args_list:
            self.assertIs(call.args[0][0], patcher)
        self.assertIs(next(iter(cache._entries.values())).model, model)

    def test_many_default_node_instances_initialize_each_model_once(self):
        from nsfw_guard import classifiers
        from nsfw_guard.nodes import NSFWGuardAIO
        image_model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[10.0, -10.0]])))
        image_classifier = Classifier(image_model,
            Mock(return_value={"pixel_values": torch.zeros(1, 3, 2, 2)}), 1, "cpu", object())
        text_classifier = Classifier(object(), object(), 1, "cpu", object())
        loader = Mock(side_effect=lambda modality, *args:
            image_classifier if modality == "image" else text_classifier)
        cache = ModelCache(loader, lambda: "cpu")
        manager = SimpleNamespace(load_models_gpu=Mock())
        with patch.object(classifiers, "_MODEL_CACHE", cache), patch.dict(
                "sys.modules", {"comfy.model_management": manager}):
            nodes = [NSFWGuardAIO() for _ in range(20)]
            self.assertTrue(all(node._backend.cache is cache for node in nodes))
            for index, node in enumerate(nodes):
                node.guard(image=torch.full((1, 2, 2, 3), index / 20))
                with node._backend.cache.use("prompt", DEFAULT_TEXT_MODEL) as classifier:
                    self.assertIs(classifier, text_classifier)
            # A later execution with newly constructed nodes still uses the cache.
            NSFWGuardAIO().guard(image=torch.ones(1, 2, 2, 3))
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(image_model.call_count, 21)
        self.assertEqual(len(cache._entries), 2)

    def test_failed_load_is_not_cached(self):
        loader = Mock(side_effect=[GuardError("model_load_failed", "failed"), object()])
        cache = ModelCache(loader, lambda: "cpu")
        with self.assertRaises(GuardError):
            with cache.use("image", "model"):
                pass
        with cache.use("image", "model"):
            pass
        self.assertEqual(loader.call_count, 2)


class LabelAndLoadingTests(unittest.TestCase):
    def config(self, labels=None, **kwargs):
        return SimpleNamespace(num_labels=2, id2label=labels or {0: "LABEL_0", 1: "LABEL_1"}, problem_type=None, **kwargs)

    def test_label_mapping_and_ambiguity(self):
        self.assertEqual(resolve_nsfw_index(self.config(), "prompt", DEFAULT_TEXT_MODEL), 0)
        for labels, expected in [({0: "normal", 1: "nsfw"}, 1), ({0: " NSFW ", 1: "SFW"}, 0)]:
            self.assertEqual(resolve_nsfw_index(self.config(labels), "image", "model"), expected)
        for modality, labels in [("image", {0: "LABEL_0", 1: "LABEL_1"}), ("prompt", {0: "LABEL_0", 1: "LABEL_1"}), ("image", {0: "nsfw", 1: "nsfw"}), ("image", {0: "nsfw", 1: "unknown"})]:
            with self.assertRaises(GuardError) as caught:
                resolve_nsfw_index(self.config(labels), modality, "unrecognized/model")
            self.assertEqual(caught.exception.payload["error"], "ambiguous_labels")
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(GuardError) as caught:
                resolve_nsfw_index(self.config(), "prompt", path)
            self.assertEqual(caught.exception.payload["error"], "ambiguous_labels")
            # Local models may use either class order; only explicit labels
            # establish which probability the guard should compare.
            self.assertEqual(resolve_nsfw_index(self.config({0: "SFW", 1: "NSFW"}), "prompt", path), 1)
        config = self.config()
        config.num_labels = 3
        with self.assertRaises(GuardError):
            resolve_nsfw_index(config, "prompt", DEFAULT_TEXT_MODEL)

    def test_non_classification_heads_are_rejected(self):
        for problem_type in ("regression", "multi_label_classification", "unknown"):
            config = self.config({0: "NSFW", 1: "SFW"})
            config.problem_type = problem_type
            with self.subTest(problem_type=problem_type):
                with self.assertRaises(GuardError) as caught:
                    resolve_nsfw_index(config, "prompt", "model")
                self.assertEqual(caught.exception.payload["error"], "unsupported_model")
        config.problem_type = "single_label_classification"
        self.assertEqual(resolve_nsfw_index(config, "prompt", "model"), 0)

    def test_local_directory_named_like_default_model_is_still_ambiguous(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as path:
            try:
                os.chdir(path)
                Path(DEFAULT_TEXT_MODEL).mkdir(parents=True)
                with self.assertRaises(GuardError) as caught:
                    resolve_nsfw_index(self.config(), "prompt", DEFAULT_TEXT_MODEL)
                self.assertEqual(caught.exception.payload["error"], "ambiguous_labels")
            finally:
                os.chdir(original_cwd)

    def test_device_cpu_and_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(select_device(), "cpu")
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.current_device", return_value=3):
            self.assertEqual(select_device(), "cuda:3")

    def test_loader_uses_only_matching_auto_classes(self):
        for modality in ("prompt", "image"):
            with self.subTest(modality=modality):
                model = Mock(config=self.config({0: "normal", 1: "nsfw"}))
                processor = Mock(is_fast=True)
                factories = {name: Mock() for name in ("AutoTokenizer", "AutoImageProcessor", "AutoModelForSequenceClassification", "AutoModelForImageClassification")}
                processor_name = "AutoTokenizer" if modality == "prompt" else "AutoImageProcessor"
                model_name = "AutoModelForSequenceClassification" if modality == "prompt" else "AutoModelForImageClassification"
                factories[processor_name].from_pretrained.return_value = processor
                factories[model_name].from_pretrained.return_value = model
                with patch.dict("sys.modules", {"transformers": SimpleNamespace(**factories)}), patch("nsfw_guard.classifiers.local_model_source", return_value="/local/model"):
                    loaded = load_classifier(modality, "model", "cpu")
                self.assertIs(loaded.model, model)
                self.assertEqual(loaded.nsfw_index, 1)
                if modality == "image":
                    factories[processor_name].from_pretrained.assert_called_once_with("/local/model", backend="pil", trust_remote_code=False, local_files_only=True)
                model.eval.assert_called_once()
                model.to.assert_called_once_with("cpu")
                factories[model_name].from_pretrained.assert_called_once_with("/local/model", use_safetensors=True, output_loading_info=True, trust_remote_code=False, local_files_only=True)
                for name, factory in factories.items():
                    if name not in (processor_name, model_name):
                        factory.from_pretrained.assert_not_called()

    def test_checkpoint_lacking_classification_head_is_rejected(self):
        for modality in ("prompt", "image"):
            with self.subTest(modality=modality):
                model = Mock(config=self.config({0: "normal", 1: "nsfw"}))
                processor = Mock(is_fast=True)
                factories = {name: Mock() for name in ("AutoTokenizer", "AutoImageProcessor", "AutoModelForSequenceClassification", "AutoModelForImageClassification")}
                processor_name = "AutoTokenizer" if modality == "prompt" else "AutoImageProcessor"
                model_name = "AutoModelForSequenceClassification" if modality == "prompt" else "AutoModelForImageClassification"
                factories[processor_name].from_pretrained.return_value = processor
                factories[model_name].from_pretrained.return_value = (model, {"missing_keys": ["classifier.weight", "classifier.bias"]})
                with patch.dict("sys.modules", {"transformers": SimpleNamespace(**factories)}), patch("nsfw_guard.classifiers.local_model_source", return_value="/local/model"):
                    with self.assertRaises(GuardError) as caught:
                        load_classifier(modality, "model", "cpu")
                self.assertEqual(caught.exception.payload["error"], "invalid_model")
                self.assertIn("classification head", caught.exception.payload["reason"])

    def test_auth_and_download_errors_do_not_expose_secrets(self):
        for status, code in [(401, "model_access_denied"), (403, "model_access_denied"), (500, "model_load_failed")]:
            for wrapper in (None, "__cause__", "__context__"):
                with self.subTest(status=status, wrapper=wrapper):
                    error = RuntimeError("hf_secret_not_to_print")
                    error.response = SimpleNamespace(status_code=status)
                    if wrapper:
                        outer = OSError("wrapped hf_secret_not_to_print")
                        setattr(outer, wrapper, error)
                        error = outer
                    factories = {name: Mock() for name in ("AutoTokenizer", "AutoImageProcessor", "AutoModelForSequenceClassification", "AutoModelForImageClassification")}
                    factories["AutoImageProcessor"].from_pretrained.side_effect = error
                    with patch.dict("sys.modules", {"transformers": SimpleNamespace(**factories)}), patch("nsfw_guard.classifiers.local_model_source", return_value="/local/model"):
                        with self.assertRaises(GuardError) as caught:
                            load_classifier("image", "model", "cpu")
                    self.assertEqual(caught.exception.payload["error"], code)
                    self.assertNotIn("hf_secret", str(caught.exception))

    def test_deberta_preserves_trained_tokenizer(self):
        config = self.config(model_type="deberta-v2")
        model = Mock(config=config)
        factories = {name: Mock() for name in ("AutoTokenizer", "AutoImageProcessor", "AutoModelForSequenceClassification", "AutoModelForImageClassification")}
        factories["AutoModelForSequenceClassification"].from_pretrained.return_value = model
        factories["AutoTokenizer"].from_pretrained.return_value = Mock(is_fast=True)
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(**factories)}), patch("nsfw_guard.classifiers.local_model_source", return_value="/local/model"):
            load_classifier("prompt", DEFAULT_TEXT_MODEL, "cpu")
        factories["AutoTokenizer"].from_pretrained.assert_called_once_with("/local/model", fix_mistral_regex=False, trust_remote_code=False, local_files_only=True)


class InferenceTests(unittest.TestCase):
    def test_image_validation(self):
        backend = Backend()
        for channels in (1, 3, 4):
            backend.validate_image(torch.rand(2, 4, 5, channels))
        # VAE decoding jitter (fp16/bf16) within [-1e-3, 1 + 1e-3] must be accepted
        backend.validate_image(torch.tensor([[[[-0.0001, 0.5, 1.0001]]]]))
        bad = [None, [], torch.zeros(4, 4, 3), torch.zeros(0, 4, 4, 3), torch.zeros(1, 0, 4, 3), torch.zeros(1, 4, 4, 2), torch.zeros(1, 4, 4, 3, dtype=torch.int32), torch.full((1, 2, 2, 3), float("nan")), torch.full((1, 2, 2, 3), float("inf")), torch.full((1, 2, 2, 3), -0.1), torch.full((1, 2, 2, 3), 1.1)]
        for image in bad:
            with self.subTest(image=type(image)):
                with self.assertRaises(GuardError) as caught:
                    backend.validate_image(image)
                self.assertEqual(caught.exception.payload["error"], "invalid_image")

    def test_logits_and_probabilities(self):
        self.assertAlmostEqual(_score(torch.tensor([[0.0, 0.0]]), 0, "prompt"), 0.5)
        self.assertGreater(_score(torch.tensor([[-10.0, 10.0]]), 1, "image"), 0.99)
        for logits in (torch.zeros(1, 3), torch.zeros(2, 2), torch.zeros(2), torch.tensor([[float("nan"), 0.0]]), torch.tensor([[float("inf"), 0.0]])):
            with self.assertRaises(GuardError):
                _score(logits, 0, "prompt")

    def test_real_tokenizer_checks_trailing_prompt_and_overlaps(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        raw = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "safe": 2, "flag": 3}, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", model_max_length=10**30)
        seen = []
        class Model:
            config = SimpleNamespace(max_position_embeddings=8)
            def __call__(self, input_ids, **kwargs):
                seen.append(input_ids.tolist()[0])
                flag = bool((input_ids == 3).any().item())
                return SimpleNamespace(logits=torch.tensor([[8.0, -8.0] if flag else [-8.0, 8.0]]))
        classifier = Classifier(Model(), tokenizer, 0, "cpu")
        backend = Backend(ModelCache(lambda *args: classifier, lambda: "cpu"))
        result = backend.classify_prompt("text", "safe " * 30 + "flag")
        self.assertGreater(result.confidence, 0.99)
        self.assertGreater(len(seen), 1)
        self.assertNotIn(3, seen[0])
        self.assertIn(3, seen[-1])
        self.assertTrue(all(len(window) <= 8 for window in seen))
        self.assertAlmostEqual(backend.classify_prompt("text", "safe").confidence, 0, places=5)

    def test_entire_image_batch_rgb_conversion_and_maximum_index(self):
        for channels in (1, 3, 4):
            with self.subTest(channels=channels):
                seen = []
                def processor(images, return_tensors):
                    seen.append(images.mode)
                    return {"pixel_values": torch.tensor([[images.getpixel((0, 0))[0] / 255.0]])}
                def model(pixel_values):
                    value = pixel_values[0, 0].item()
                    return SimpleNamespace(logits=torch.tensor([[1.0 - value, value]]) * 10)
                classifier = Classifier(model, processor, 1, "cpu")
                backend = Backend(ModelCache(lambda *args: classifier, lambda: "cpu"))
                image = torch.zeros(3, 2, 2, channels)
                image[1] = 0.7
                image[2] = 1.0
                original = image.clone()
                backend.validate_image(image)
                result = backend.classify_image("image", image)
                self.assertEqual(result.index, 2)
                self.assertGreater(result.confidence, 0.99)
                self.assertEqual(seen, ["RGB"] * 3)
                self.assertTrue(torch.equal(image, original))

    def test_inference_failure_and_interrupt(self):
        processor = Mock(side_effect=RuntimeError("private input data"))
        classifier = Classifier(Mock(), processor, 1, "cpu")
        backend = Backend(ModelCache(lambda *args: classifier, lambda: "cpu"))
        with self.assertRaises(GuardError) as caught:
            backend.classify_image("image", torch.zeros(1, 2, 2, 3))
        self.assertEqual(caught.exception.payload["error"], "inference_failed")
        self.assertNotIn("private", str(caught.exception))
        class InterruptProcessingException(Exception):
            pass
        processor.side_effect = InterruptProcessingException()
        with self.assertRaises(InterruptProcessingException):
            backend.classify_image("image", torch.zeros(1, 2, 2, 3))

    def test_local_vit_checkpoint_with_real_processor_and_model(self):
        from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

        config = ViTConfig(image_size=16, patch_size=8, hidden_size=16, num_hidden_layers=1,
                           num_attention_heads=2, intermediate_size=32, num_labels=2,
                           id2label={0: "normal", 1: "nsfw"}, label2id={"normal": 0, "nsfw": 1})
        with tempfile.TemporaryDirectory() as path:
            ViTForImageClassification(config).save_pretrained(path)
            ViTImageProcessor(size={"height": 16, "width": 16}).save_pretrained(path)
            backend = Backend(ModelCache(device_selector=lambda: "cpu"))
            image = torch.rand(2, 16, 16, 3)
            backend.validate_image(image)
            assessment = backend.classify_image(path, image)
            self.assertTrue(0 <= assessment.confidence <= 1)
            self.assertIn(assessment.index, (0, 1))

    def test_local_text_model_requires_labels_and_respects_reversed_order(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
        from nsfw_guard.errors import NSFWDetectedError
        from nsfw_guard.nodes import NSFWGuardAIO

        raw = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "content": 2}, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", model_max_length=16)
        model = BertForSequenceClassification(BertConfig(
            vocab_size=3, hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16, max_position_embeddings=16, num_labels=2,
        ))
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.copy_(torch.tensor([-10.0, 10.0]))
        with tempfile.TemporaryDirectory() as path:
            tokenizer.save_pretrained(path)
            model.save_pretrained(path)
            node = NSFWGuardAIO(Backend(ModelCache(device_selector=lambda: "cpu")))
            with self.assertRaises(GuardError) as caught:
                node.guard(prompt="content", model_text_classification=path)
            self.assertEqual(caught.exception.payload["error"], "ambiguous_labels")

            model.config.id2label = {0: "SFW", 1: "NSFW"}
            model.config.label2id = {"SFW": 0, "NSFW": 1}
            model.save_pretrained(path)
            with self.assertRaises(NSFWDetectedError) as caught:
                node.guard(prompt="content", model_text_classification=path)
            self.assertGreater(caught.exception.payload["confidence"], 0.99)

    def test_context_length_ignores_tokenizer_sentinel(self):
        self.assertEqual(_context_length(SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)), SimpleNamespace(model_max_length=10**30)), 512)
        self.assertEqual(_context_length(SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)), SimpleNamespace(model_max_length=128)), 128)

    def test_context_length_allows_large_context(self):
        self.assertEqual(_context_length(SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048)), SimpleNamespace(model_max_length=2048)), 2048)

    def test_oom_error_is_preserved_and_soft_empty_cache_called(self):
        oom_err = torch.cuda.OutOfMemoryError("CUDA out of memory")
        processor = Mock(side_effect=oom_err)
        classifier = Classifier(Mock(), processor, 1, "cpu")
        backend = Backend(ModelCache(lambda *args: classifier, lambda: "cpu"))
        mock_mm = SimpleNamespace(is_oom=lambda ex: True, soft_empty_cache=Mock())
        with patch.dict("sys.modules", {"comfy": SimpleNamespace(model_management=mock_mm), "comfy.model_management": mock_mm}):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                backend.classify_image("image", torch.zeros(1, 2, 2, 3))
        mock_mm.soft_empty_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class ModelDirectoryTests(unittest.TestCase):
    def test_download_to_comfy_models_and_reuse_without_hub(self):
        from nsfw_guard.classifiers import local_model_source
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "nsfw_guard" / "org" / "classifier"
            def download(**kwargs):
                Path(kwargs["local_dir"]).mkdir(parents=True)
            with patch.dict("sys.modules", {"folder_paths": SimpleNamespace(models_dir=root)}), patch("huggingface_hub.snapshot_download", side_effect=download) as hub, patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}):
                self.assertEqual(local_model_source("org/classifier"), str(target))
                self.assertEqual(hub.call_args.kwargs["local_dir"], str(target))
                self.assertNotIn("cache_dir", hub.call_args.kwargs)
                with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
                    self.assertEqual(local_model_source("org/classifier"), str(target))
                hub.assert_called_once()

    def test_local_directory_never_downloads(self):
        from nsfw_guard.classifiers import local_model_source
        with tempfile.TemporaryDirectory() as root, patch("huggingface_hub.snapshot_download") as hub:
            self.assertEqual(local_model_source(root), str(Path(root).resolve()))
            hub.assert_not_called()

    def test_offline_missing_model_and_unsafe_id_do_not_download(self):
        from nsfw_guard.classifiers import local_model_source
        with tempfile.TemporaryDirectory() as root, patch.dict("sys.modules", {"folder_paths": SimpleNamespace(models_dir=root)}), patch("huggingface_hub.snapshot_download") as hub:
            with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
                with self.assertRaises(OSError):
                    local_model_source("org/missing")
            with self.assertRaises(ValueError):
                local_model_source("../escape")
            hub.assert_not_called()

    def test_failed_download_is_retried(self):
        from nsfw_guard.classifiers import local_model_source
        with tempfile.TemporaryDirectory() as root, patch.dict("sys.modules", {"folder_paths": SimpleNamespace(models_dir=root)}), patch("huggingface_hub.snapshot_download", side_effect=OSError("interrupted")) as hub, patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}):
            for _ in range(2):
                with self.assertRaises(OSError):
                    local_model_source("org/classifier")
            self.assertEqual(hub.call_count, 2)
            self.assertFalse((Path(root) / "nsfw_guard/org/classifier/.download_complete").exists())
