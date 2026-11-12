"""Use the bootstrapped CorridorKey profile from any checkout location."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKILL = ROOT / ".agents/skills/corridorkey"


def load_runtime(engine="v1"):
    path = ROOT / (".local/runtime-v2.json" if engine == "v2" else ".local/runtime.json")
    if not path.is_file():
        raise RuntimeError(f"Run python bootstrap.py --engine {engine} --profile "
                           + ("arc" if engine == "v2" else "p600|arc|cuda|cpu") + " on this device first.")
    config = json.loads(path.read_text(encoding="utf-8"))
    if engine == "v2":
        if config.get("engine") != "v2" or config.get("profile") != "arc":
            raise RuntimeError("Invalid CK2 runtime profile. Rerun v2 bootstrap; no other engine selected.")
        requested = config.get("device", "")
        if requested != "xpu" and not (requested.startswith("xpu:") and requested[4:].isdigit()):
            raise RuntimeError("The CK2 runtime requires an explicit XPU device; rerun v2 bootstrap.")
    for key in ("python", "ffmpeg", "ffprobe"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else ROOT / value)
        if not Path(config[key]).is_file():
            raise RuntimeError(f"The configured {key} is missing. Rerun bootstrap on this device.")
    if engine == "v2":
        source = Path(config["source"])
        source = source if source.is_absolute() else ROOT / source
        if not (source / "infer.py").is_file():
            raise RuntimeError("The CK2 source is missing. Rerun v2 bootstrap on this device.")
        expected = hashlib.sha256((ROOT / "integrations/ck2/xpu.patch").read_bytes()).hexdigest()
        if config.get("patch_sha256") != expected:
            raise RuntimeError("The CK2 integration patch changed. Rerun v2 bootstrap before using it.")
        config["source"] = str(source)
    return config


def option_value(rest, option, default=None):
    result = default
    for i, arg in enumerate(rest):
        if arg.startswith(option + "="):
            result = arg.split("=", 1)[1]
        elif arg == option and i + 1 < len(rest):
            result = rest[i + 1]
    return result


def v2_command(config, action, rest):
    prefix = [config["python"], "-X", "utf8"]
    if action == "check":
        if option_value(rest, "--device") is not None:
            raise ValueError("Select the v2 device with bootstrap --device xpu:N.")
        return [*prefix, str(ROOT / "integrations/ck2/check_xpu.py"),
                "--source", config["source"], "--device", config["device"], *rest]
    if action == "verify":
        if option_value(rest, "--device") is not None:
            raise ValueError("Select the v2 device with bootstrap --device xpu:N.")
        defaults = [] if option_value(rest, "--output") else ["--output", str(ROOT / "Output/ck2-xpu-smoke")]
        return [*prefix, str(ROOT / "integrations/ck2/verify.py"), "--source", config["source"],
                "--device", config["device"], *defaults, *rest]
    if any(arg == "--device" or arg.startswith("--device=") for arg in rest):
        raise ValueError("Select the v2 device with bootstrap --device xpu:N, then rerun the command.")
    defaults = ["--device", config["device"]]
    for option, value in (("--ffmpeg_bin", config["ffmpeg"]), ("--output_format", "exr")):
        if option_value(rest, option) is None:
            defaults += [option, value]
    output = option_value(rest, "--output_dir")
    if output and option_value(rest, "--temp_dir") is None:
        defaults += ["--temp_dir", str(Path(output).resolve() / ".ck2-temp")]
    if option_value(rest, "--output_format", "exr") == "video":
        outputs = []
        for i, arg in enumerate(rest):
            if arg.startswith("--outputs="):
                outputs += arg.split("=", 1)[1].split(",")
            elif arg == "--outputs":
                for item in rest[i + 1:]:
                    if item.startswith("--"):
                        break
                    outputs += item.split(",")
        if "cutout" in [part.strip().lower() for part in outputs]:
            raise ValueError("Use --output_format exr for transparent cutout. The upstream video cutout "
                             "writer selects software ProRes; it is not a QSV alpha export.")
        codec = option_value(rest, "--ffmpeg_codec", config["encoder"])
        if codec.endswith(("_qsv", "_nvenc")) and option_value(rest, "--crf") is not None:
            raise ValueError("Use --bitrate with the hardware encoder, without --crf.")
        for option, value in (("--ffmpeg_codec", config["encoder"]), ("--pix_fmt", "nv12"),
                              ("--bitrate", "12M")):
            if option_value(rest, option) is None:
                defaults += [option, value]
    return [*prefix, str(Path(config["source"]) / "infer.py"), *defaults, *rest]


def build_command(config, action, rest):
    if config.get("engine") == "v2" and action in ("check", "keyer", "verify"):
        return v2_command(config, action, rest)
    if action == "verify":
        raise ValueError("verify is a v2 action. Use scripts/verify_inference.py for v1.")
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
    parser.add_argument("--engine", choices=("v1", "v2"), default="v1",
                        help="v1 remains the default; v2 is the experimental Arc runtime")
    parser.add_argument("action", choices=("check", "hint", "green", "keyer", "verify"))
    args, rest = parser.parse_known_args()
    config = load_runtime(args.engine)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    if config["profile"] == "cpu":
        environment["CORRIDORKEY_SKIP_COMPILE"] = "1"
    command = build_command(config, args.action, rest)
    result = subprocess.run(command, cwd=ROOT, env=environment)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
