"""Run the complete Hull GeoSMILE analysis and visualisation pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICE = ROOT / "Police.uk.zip"
OUTPUT = ROOT / "outputs"


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    if not POLICE.exists():
        raise SystemExit(f"Missing police data: {POLICE}")
    py = sys.executable
    run([py, "geosmile_hull.py", "--police-data", str(POLICE), "--output", str(OUTPUT)])
    run([py, "make_visuals_hull.py"])
    run([py, "make_figures_hull.py"])
    run([py, "write_interpretation.py"])
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
