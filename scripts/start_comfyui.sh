#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ "${1:-}" == "--install" ]]; then
    UV_CACHE_DIR=/tmp/uv-cache uv pip install --python "$repo_dir/.venv/bin/python" -r "$repo_dir/.comfyui/requirements.txt" -r "$repo_dir/requirements.txt"
    shift
fi
exec "$repo_dir/.venv/bin/python" "$repo_dir/.comfyui/main.py" --cpu --listen 127.0.0.1 --port 8188 "$@"
