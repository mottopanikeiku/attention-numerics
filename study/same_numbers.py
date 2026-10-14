"""Check regenerated JSON reports against the committed copies, allowing last-bit float noise.

Norm reductions can round differently on different CPUs, so a byte comparison of these
reports fails on some CI runners even though every value agrees to about 15 digits.

python -m study.same_numbers results/hardware/summary.json
"""

import json
import math
import subprocess
import sys
from pathlib import Path


def differences(committed, regenerated, where: str = "$") -> list[str]:
    if isinstance(committed, float) or isinstance(regenerated, float):
        pair = (committed, regenerated)
        numbers = all(isinstance(v, int | float) and not isinstance(v, bool) for v in pair)
        if numbers and math.isclose(committed, regenerated, rel_tol=1e-12, abs_tol=1e-15):
            return []
        return [f"{where}: {committed!r} != {regenerated!r}"]
    if isinstance(committed, dict) and isinstance(regenerated, dict):
        if committed.keys() != regenerated.keys():
            return [f"{where}: keys differ"]
        return [
            d
            for key in committed
            for d in differences(committed[key], regenerated[key], f"{where}.{key}")
        ]
    if isinstance(committed, list) and isinstance(regenerated, list):
        if len(committed) != len(regenerated):
            return [f"{where}: {len(committed)} items != {len(regenerated)}"]
        return [
            d
            for i, (a, b) in enumerate(zip(committed, regenerated, strict=True))
            for d in differences(a, b, f"{where}[{i}]")
        ]
    return [] if committed == regenerated else [f"{where}: {committed!r} != {regenerated!r}"]


def main() -> None:
    problems = []
    for path in sys.argv[1:]:
        committed = subprocess.run(
            ["git", "show", f"HEAD:{path}"], check=True, capture_output=True, text=True
        ).stdout
        regenerated = Path(path).read_text()
        found = differences(json.loads(committed), json.loads(regenerated))
        problems += [f"{path} {d}" for d in found]
    if problems:
        print("\n".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
