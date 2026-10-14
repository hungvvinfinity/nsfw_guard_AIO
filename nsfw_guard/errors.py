"""Machine-readable exceptions exposed by ComfyUI's execution_error event."""

import json


class GuardError(Exception):
    def __init__(self, error: str, reason: str, modality: str | None = None):
        self.payload = {"error": error, "reason": reason}
        if modality is not None:
            self.payload["type"] = modality
        super().__init__(json.dumps(self.payload, allow_nan=False))


class NSFWDetectedError(Exception):
    def __init__(self, modality: str, confidence: float, index: int | None = None):
        reason = "Prompt contains NSFW content"
        if modality == "image":
            reason = f"Image at batch index {index} contains NSFW content"
        self.payload = {"reason": reason, "confidence": confidence, "type": modality}
        super().__init__(json.dumps(self.payload, allow_nan=False))
