"""Prepare a cloned CorridorKey checkout for the selected machine, entirely locally."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEVICES = {"p600": "cuda", "arc": "xpu", "cuda": "cuda", "cpu": "cpu"}


def python312_command():
    if sys.version_info[:2] == (3, 12):
        return [sys.executable]
    direct = shutil.which("python3.12")
    if direct:
        return [direct]
    if sys.platform == "win32" and shutil.which("py"):
        check = subprocess.run(["py", "-3.12", "-c", "pass"], capture_output=True)
        if check.returncode == 0:
            return ["py", "-3.12"]
    if shutil.which("uv"):
        return ["uv", "run", "--no-project", "--python", "3.12", "python"]
    raise RuntimeError("Install Python 3.12 or uv, then rerun bootstrap. System Python will not be replaced.")


def build_plan(profile, with_tests=False):
    setup = [*python312_command(), "-X", "utf8", str(ROOT / "scripts/setup_gpu.py"), "--profile", profile]
    if with_tests:
        setup.append("--with-tests")
    return {
        "root": str(ROOT),
        "profile": profile,
        "device": DEVICES[profile],
        "setup_command": setup,
        "skill": str(ROOT / ".agents/skills/corridorkey/SKILL.md"),
        "runtime_file": str(ROOT / ".local/runtime.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=DEVICES, required=True)
    parser.add_argument("--with-tests", action="store_true")
    parser.add_argument("--ffprobe", type=Path, help="Use an existing ffprobe executable")
    parser.add_argument("--skip-model", action="store_true", help="Download the green checkpoint on first use instead")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the setup plan without changing files or installing"
    )
    args = parser.parse_args()
    if sys.platform not in ("win32", "linux"):
        parser.error("These setup profiles target Windows/Linux; this platform has not been provisioned or validated.")
    plan = build_plan(args.profile, args.with_tests)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    subprocess.run(plan["setup_command"], cwd=ROOT, check=True)
    python = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    configure = [str(python), "-X", "utf8", str(ROOT / "scripts/configure_runtime.py"), "--profile", args.profile]
    if args.ffprobe:
        configure += ["--ffprobe", str(args.ffprobe.resolve())]
    if args.skip_model:
        configure.append("--skip-model")
    subprocess.run(configure, cwd=ROOT, check=True)
    print("Ready. Open this folder in Codex and use $corridorkey. Run python run.py --help for local commands.")


if __name__ == "__main__":
    main()
