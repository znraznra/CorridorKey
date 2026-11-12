"""Build an instruction/helper ZIP for hosts that accept uploaded skill bundles."""

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".agents/skills/corridorkey"


def package(output):
    output = Path(output).resolve()
    if output.suffix.lower() != ".zip" or output.is_relative_to(SKILL.resolve()):
        raise ValueError("Choose a .zip output outside the source skill directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = [p for p in SKILL.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in entries:
            bundle.write(path, "corridorkey/" + path.relative_to(SKILL).as_posix())
        bundle.write(ROOT / "LICENSE", "corridorkey/LICENSE")
    with zipfile.ZipFile(output) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Skill ZIP verification failed.")
    return {"path": str(output), "files": len(entries) + 1, "contains_engine_or_weights": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/dist/corridorkey.zip")
    args = parser.parse_args()
    print(json.dumps(package(args.output), indent=2))


if __name__ == "__main__":
    main()
