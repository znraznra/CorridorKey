#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if command -v "${PYTHON312:-python3.12}" >/dev/null 2>&1; then
    "${PYTHON312:-python3.12}" scripts/setup_gpu.py --profile arc "$@"
elif command -v uv >/dev/null 2>&1; then
    uv run --no-project --python 3.12 python scripts/setup_gpu.py --profile arc "$@"
else
    echo 'Python 3.12 or uv is required. Set PYTHON312 to your Python 3.12 executable.' >&2
    exit 1
fi
