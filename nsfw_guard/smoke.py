"""Opt-in real-model checks: python -m nsfw_guard.smoke --modality prompt."""

import argparse
import json

from .classifiers import Backend, DEFAULT_IMAGE_MODEL, DEFAULT_TEXT_MODEL
from .errors import GuardError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("prompt", "image", "both"), default="prompt")
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    args = parser.parse_args()
    backend = Backend()
    try:
        if args.modality in ("prompt", "both"):
            assessment = backend.classify_prompt(args.text_model, "A peaceful mountain landscape at sunrise")
            print(json.dumps({"type": "prompt", "nsfw_probability": assessment.confidence}))
        if args.modality in ("image", "both"):
            import torch

            image = torch.full((1, 224, 224, 3), 0.5)
            backend.validate_image(image)
            assessment = backend.classify_image(args.image_model, image)
            print(json.dumps({"type": "image", "nsfw_probability": assessment.confidence}))
    except GuardError as exc:
        print(str(exc))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
