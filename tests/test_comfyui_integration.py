"""Opt in with COMFYUI_URL after installing the node in a running ComfyUI."""

import json
import os
import time
import unittest
from urllib.request import Request, urlopen

from nsfw_guard.classifiers import DEFAULT_TEXT_MODEL


@unittest.skipUnless(os.environ.get("COMFYUI_URL"), "COMFYUI_URL is not set")
class ComfyUIIntegrationTests(unittest.TestCase):
    def test_detection_stops_dependent_node_with_json(self):
        base = os.environ["COMFYUI_URL"].rstrip("/")
        model = os.environ.get("COMFYUI_GUARD_TEXT_MODEL", DEFAULT_TEXT_MODEL)
        workflow = {
            "1": {"class_type": "NSFWGuardAIO", "inputs": {
                "model_text_classification": model,
                "prompt": "A peaceful mountain landscape",
                "prompt_sensitivity": 1.0, "image_sensitivity": 0.5,
            }},
            "2": {"class_type": "NSFWGuardAIO", "inputs": {
                "model_text_classification": model,
                "prompt": ["1", 0],
                "prompt_sensitivity": 0.5, "image_sensitivity": 0.5,
            }},
        }
        request = Request(base + "/prompt", data=json.dumps({"prompt": workflow}).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            queued = json.load(response)
        self.assertNotIn("error", queued)
        prompt_id = queued["prompt_id"]
        deadline = time.monotonic() + 300
        record = None
        while time.monotonic() < deadline:
            with urlopen(base + "/history/" + prompt_id, timeout=30) as response:
                record = json.load(response).get(prompt_id)
            if record is not None:
                break
            time.sleep(0.25)
        self.assertIsNotNone(record, "Workflow did not finish within 300 seconds")
        errors = [data for name, data in record["status"]["messages"] if name == "execution_error"]
        self.assertEqual(len(errors), 1)
        event = errors[0]
        self.assertTrue(event["exception_type"].endswith("NSFWDetectedError"), event["exception_message"])
        self.assertEqual(event["node_id"], "1")
        self.assertNotIn("2", event["executed"])
        self.assertNotIn("2", record["outputs"])
        payload = json.loads(event["exception_message"].strip())
        self.assertEqual(set(payload), {"reason", "confidence", "type"})
        self.assertEqual(payload["type"], "prompt")
        self.assertTrue(0 <= payload["confidence"] <= 1)


if __name__ == "__main__":
    unittest.main()
