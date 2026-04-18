"""Cheap hyperparameter sweep (same seed, train CSV). Run from repo root: PYTHONPATH=src python scripts/quick_sweep.py"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
EXP_ROOT = ROOT / "artifacts" / "quick_sweep"


def run_one(name: str, extra: list[str]) -> dict:
    out = EXP_ROOT / name
    if out.exists():
        shutil.rmtree(out)
    cmd = [
        PY,
        str(ROOT / "scripts" / "train.py"),
        "--artifacts-dir",
        str(out),
        "--seed",
        "42",
        "--device",
        "cpu",
        *extra,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:] if r.stdout else "")
        print(r.stderr[-2000:] if r.stderr else "")
        raise RuntimeError(f"Train failed for {name}: exit {r.returncode}")
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    return {
        "name": name,
        "mse": float(meta["cv_mean_mse_normalized"]),
        "mae": float(meta["cv_mean_mae_normalized"]),
        "args": extra,
    }


def pick_winner(rows: list[dict], mae_slip: float = 1.025) -> dict:
    base = next(r for r in rows if r["name"] == "base")
    b_mae = base["mae"]
    cap = b_mae * mae_slip
    feasible = [r for r in rows if r["mae"] <= cap]
    if feasible:
        return min(feasible, key=lambda r: r["mse"])
    return min(rows, key=lambda r: r["mse"] + 0.25 * r["mae"])


def main() -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, list[str]]] = [
        ("base", []),
        ("hybrid095", ["--hybrid-mse-weight", "0.95"]),
        ("hybrid096", ["--hybrid-mse-weight", "0.96"]),
        ("hybrid098", ["--hybrid-mse-weight", "0.98"]),
        ("cosine", ["--cosine-scheduler"]),
        ("adamw_wd4", ["--optimizer", "adamw", "--weight-decay", "0.0001"]),
        ("adamw_cos", ["--optimizer", "adamw", "--weight-decay", "0.0001", "--cosine-scheduler"]),
        ("hybrid096_cos", ["--hybrid-mse-weight", "0.96", "--cosine-scheduler"]),
    ]
    rows: list[dict] = []
    for name, extra in specs:
        print(f"=== {name} ===", flush=True)
        rows.append(run_one(name, extra))

    rows.sort(key=lambda r: (r["mse"], r["mae"]))
    mae_slip = 1.025
    winner = pick_winner(rows, mae_slip=mae_slip)
    summary = {"runs": rows, "winner": winner, "mae_slip_used": mae_slip}
    (EXP_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSorted by MSE:")
    for r in sorted(rows, key=lambda x: x["mse"]):
        mark = " *" if r["name"] == winner["name"] else ""
        print(f"  {r['name']:16} MSE={r['mse']:.6f} MAE={r['mae']:.6f}{mark}")
    print(f"\nPicked: {winner['name']} (MAE <= base * {mae_slip:.3f} when choosing among feasible)")
    print(json.dumps(winner, indent=2))


if __name__ == "__main__":
    main()
