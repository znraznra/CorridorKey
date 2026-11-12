#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    echo 'Run bash setup_arc_linux.sh first.' >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    set -- --help
fi
exec .venv/bin/python -X utf8 corridorkey_cli.py --device "${CORRIDORKEY_DEVICE:-xpu}" "$@"
