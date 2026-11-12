"""Use the bootstrapped CorridorKey profile from any checkout location."""

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKILL = ROOT / ".agents/skills/corridorkey"


def load_runtime():
    path = ROOT / ".local/runtime.json"
    if not path.is_file():
        raise RuntimeError("Run python bootstrap.py --profile p600|arc|cuda|cpu on this device first.")
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("python", "ffmpeg", "ffprobe"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else ROOT / value)
        if not Path(config[key]).is_file():
            raise RuntimeError(f"The configured {key} is missing. Rerun bootstrap on this device.")
    return config


def build_command(config, action, rest):
    prefix = [config["python"], "-X", "utf8"]
    if action == "check":
        return [*prefix, str(ROOT / "scripts/gpu_check.py"), "--device", config["device"], *rest]
    if action == "hint":
        return [*prefix, str(SKILL / "scripts/make_chroma_hint.py"), *rest]
    if action == "green":
        defaults = []
        for option, key in (("--ffmpeg", "ffmpeg"), ("--ffprobe", "ffprobe"), ("--encoder", "encoder")):
            if not any(arg == option or arg.startswith(option + "=") for arg in rest):
                defaults += [option, config[key]]
        return [*prefix, str(SKILL / "scripts/compose_green.py"), *defaults, *rest]
    return [*prefix, str(ROOT / "corridorkey_cli.py"), "--device", config["device"], *rest]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "hint", "green", "keyer"))
    args, rest = parser.parse_known_args()
    config = load_runtime()
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    if config["profile"] == "cpu":
        environment["CORRIDORKEY_SKIP_COMPILE"] = "1"
    result = subprocess.run(build_command(config, args.action, rest), cwd=ROOT, env=environment)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
