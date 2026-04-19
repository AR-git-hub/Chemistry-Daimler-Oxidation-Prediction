"""Rank artifact folders by CV metrics in metadata.json (pick seeds / runs to stack)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="List artifacts/*/metadata.json sorted by CV.")
    ap.add_argument("--artifacts-root", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    rows: list[tuple[float, float, str, str | None]] = []
    root: Path = args.artifacts_root
    if not root.is_dir():
        print("No artifacts root:", root, file=sys.stderr)
        sys.exit(1)
    for meta in root.glob("*/metadata.json"):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except OSError:
            continue
        if "cv_mean_mse_normalized" not in m:
            continue
        mse = float(m["cv_mean_mse_normalized"])
        mae = float(m.get("cv_mean_mae_normalized", float("nan")))
        seed = m.get("train_seed")
        rows.append((mse, mae, meta.parent.name, str(seed) if seed is not None else None))

    rows.sort(key=lambda r: (r[0], r[1]))
    print(f"{'cv_mse':>12} {'cv_mae':>12} {'seed':>6}  dir")
    for mse, mae, name, seed in rows[: int(args.top)]:
        print(f"{mse:12.6f} {mae:12.6f} {seed or '':>6}  {name}")


if __name__ == "__main__":
    main()
