"""Sweep hidden widths (CV only via full train.py). Run from repo root: PYTHONPATH=src python scripts/hidden_width_sweep.py"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> None:
    out_root = ROOT / "artifacts" / "hidden_width_sweep"
    out_root.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, list[str]]] = [
        ("h64", ["--hidden-dim", "64"]),
        ("h88", ["--hidden-dim", "88"]),
        ("h96", ["--hidden-dim", "96"]),
        ("h112", ["--hidden-dim", "112"]),
        ("h128_rho72", ["--hidden-dim", "128", "--rho-hidden-dim", "72"]),
        ("h128_enc88_rho72", ["--hidden-dim", "128", "--encoder-hidden-dim", "88", "--rho-hidden-dim", "72"]),
    ]
    rows: list[dict] = []
    for name, extra in specs:
        dest = out_root / name
        if dest.exists():
            shutil.rmtree(dest)
        cmd = [
            PY,
            str(ROOT / "scripts" / "train.py"),
            "--artifacts-dir",
            str(dest),
            "--device",
            "cpu",
            "--seed",
            "42",
            *extra,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        print("===", name, "===", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2500:] if r.stderr else "", r.stdout[-1500:] if r.stdout else "")
            raise RuntimeError(f"Train failed {name}")
        meta = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
        vm = meta["validation_metrics"]
        gap = sum(float(m["train_mse_normalized"]) - float(m["val_mse_normalized"]) for m in vm) / len(vm)
        rows.append(
            {
                "name": name,
                "hidden_dim": meta.get("hidden_dim"),
                "encoder_hidden_dim": meta.get("encoder_hidden_dim"),
                "rho_hidden_dim": meta.get("rho_hidden_dim"),
                "cv_mse": float(meta["cv_mean_mse_normalized"]),
                "cv_mae": float(meta["cv_mean_mae_normalized"]),
                "mean_train_minus_val_mse": gap,
            }
        )
    rows.sort(key=lambda r: (r["cv_mse"], r["cv_mae"]))
    summary = {"runs": rows, "best_mse": rows[0]}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["best_mse"], indent=2))


if __name__ == "__main__":
    main()
