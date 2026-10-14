"""Run against a local ComfyUI checkout without starting an HTTP server."""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("COMFYUI_PATH"), "COMFYUI_PATH is not set")
class ComfyExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_sys_path = list(sys.path)
        sys.path.insert(0, os.environ["COMFYUI_PATH"])
        from comfy.cli_args import args
        cls._orig_args_cpu = getattr(args, "cpu", False)
        args.cpu = True
        import execution
        import folder_paths
        import nodes
        from PIL import Image
        from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

        cls.execution = execution
        cls.nodes = nodes
        root = Path(__file__).resolve().parents[1]
        if not asyncio.run(nodes.load_custom_node(str(root))):
            raise RuntimeError("ComfyUI failed to register the node")
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.image_model = str(Path(cls.temp.name) / "vit")
        model = ViTForImageClassification(ViTConfig(
            image_size=16, patch_size=8, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, num_labels=2,
            id2label={0: "normal", 1: "nsfw"}, label2id={"normal": 0, "nsfw": 1},
        ))
        model.save_pretrained(cls.image_model)
        ViTImageProcessor(size={"height": 16, "width": 16}).save_pretrained(cls.image_model)

        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

        cls.text_model = str(Path(cls.temp.name) / "bert")
        t_model = BertForSequenceClassification(BertConfig(
            vocab_size=5, hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16, max_position_embeddings=16, num_labels=2,
            id2label={0: "SFW", 1: "NSFW"}, label2id={"SFW": 0, "NSFW": 1},
        ))
        t_model.save_pretrained(cls.text_model)
        raw = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "safe": 2, "text": 3}, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", model_max_length=16).save_pretrained(cls.text_model)

        input_dir = Path(cls.temp.name) / "input"
        output_dir = Path(cls.temp.name) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        Image.new("RGB", (16, 16), "gray").save(input_dir / "test.png")
        cls.output_dir = output_dir
        folder_paths.set_input_directory(str(input_dir))
        folder_paths.set_output_directory(str(output_dir))

    def execute(self, workflow):
        validation = asyncio.run(self.execution.validate_prompt("guard-test", workflow, None))
        self.assertTrue(validation[0], validation)
        class EventSink:
            client_id = None
            last_node_id = None
            sockets_metadata = {}
            def __init__(self):
                self.events = []
            def send_sync(self, event, data, sid=None):
                self.events.append((event, data))
        sink = EventSink()
        executor = self.execution.PromptExecutor(
            sink, cache_type=self.execution.CacheType.CLASSIC,
            cache_args={"lru": 0, "ram": 0, "ram_inactive": 0},
        )
        executor.execute(workflow, "guard-test", {"client_id": "test"}, validation[2])
        return executor, sink.events

    def image_workflow(self, sensitivity):
        return {
            "1": {"class_type": "LoadImage", "inputs": {"image": "test.png"}},
            "2": {"class_type": "NSFWGuardAIO", "inputs": {
                "model_image_classification": self.image_model, "image": ["1", 0],
                "prompt_sensitivity": 0.5, "image_sensitivity": sensitivity,
            }},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 1], "filename_prefix": "checked"}},
        }

    def test_safe_image_reaches_save_node(self):
        executor, events = self.execute(self.image_workflow(0.0))
        self.assertTrue(executor.success)
        self.assertFalse([data for event, data in events if event == "execution_error"])
        self.assertIn("3", executor.history_result["outputs"])
        output = executor.history_result["outputs"]["2"]
        self.assertEqual(json.loads(output["text"][0]), {"status": "safe"})

    def test_blocked_image_never_reaches_save_node(self):
        executor, events = self.execute(self.image_workflow(1.0))
        self.assertFalse(executor.success)
        errors = [data for event, data in events if event == "execution_error"]
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0]["exception_type"].endswith("NSFWDetectedError"))
        self.assertEqual(json.loads(errors[0]["exception_message"].strip())["type"], "image")
        self.assertNotIn("3", errors[0]["executed"])
        self.assertNotIn("3", executor.history_result["outputs"])

    def test_invalid_matching_model_fails_before_downstream(self):
        workflow = self.image_workflow(0.0)
        workflow["2"]["inputs"]["model_image_classification"] = 123
        executor, events = self.execute(workflow)
        errors = [data for event, data in events if event == "execution_error"]
        self.assertFalse(executor.success)
        self.assertEqual(json.loads(errors[0]["exception_message"].strip())["error"], "invalid_model")
        self.assertNotIn("3", errors[0]["executed"])

    @unittest.skipUnless(os.environ.get("COMFYUI_GUARD_TEXT_MODEL"), "COMFYUI_GUARD_TEXT_MODEL is not set")
    def test_requested_text_model_blocks_dependent_guard(self):
        workflow = {
            "1": {"class_type": "NSFWGuardAIO", "inputs": {
                "model_text_classification": os.environ["COMFYUI_GUARD_TEXT_MODEL"],
                "prompt": "A peaceful mountain landscape", "prompt_sensitivity": 1.0, "image_sensitivity": 0.5,
            }},
            "2": {"class_type": "NSFWGuardAIO", "inputs": {
                "model_text_classification": os.environ["COMFYUI_GUARD_TEXT_MODEL"],
                "prompt": ["1", 0], "prompt_sensitivity": 0.0, "image_sensitivity": 0.5,
            }},
        }
        executor, events = self.execute(workflow)
        errors = [data for event, data in events if event == "execution_error"]
        self.assertFalse(executor.success)
        self.assertTrue(errors[0]["exception_type"].endswith("NSFWDetectedError"))
        self.assertEqual(json.loads(errors[0]["exception_message"].strip())["type"], "prompt")
        self.assertNotIn("2", errors[0]["executed"])


    def test_prompt_only_blocks_downstream_image_without_crashing_graph(self):
        workflow = {
            "1": {"class_type": "NSFWGuardAIO", "inputs": {
                "model_text_classification": self.text_model,
                "prompt": "safe text",
                "prompt_sensitivity": 0.0,
                "image_sensitivity": 0.5,
            }},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 1], "filename_prefix": "blocked"}},
        }
        executor, events = self.execute(workflow)
        self.assertTrue(executor.success)
        errors = [data for event, data in events if event == "execution_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["exception_type"], "ExecutionBlocked")
        self.assertIn("Execution Blocked: NSFW Guard: No image provided", errors[0]["exception_message"])
        self.assertEqual(errors[0]["traceback"], [])
        # Node 1 executed successfully and outputted safe prompt
        self.assertIn("1", executor.history_result["outputs"])
        # Node 2 was cleanly blocked by ExecutionBlocker, never executed, and no unhandled crash occurred
        self.assertNotIn("2", executor.history_result["outputs"])

    @classmethod
    def tearDownClass(cls):
        sys.path[:] = cls._orig_sys_path
        try:
            from comfy.cli_args import args
            args.cpu = cls._orig_args_cpu
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
