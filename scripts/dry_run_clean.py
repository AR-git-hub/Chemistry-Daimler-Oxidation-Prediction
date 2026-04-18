"""Run inference in an isolated temporary folder to verify reproducibility."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import os
from pathlib import Path


def run() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    required_paths = [
        "src",
        "scripts",
        "artifacts",
        "data/merged/test_full.csv",
        "notebooks/inference.ipynb",
    ]
    with tempfile.TemporaryDirectory(prefix="dot_clean_run_") as tmpdir:
        tmp_root = Path(tmpdir) / "workspace"
        tmp_root.mkdir(parents=True, exist_ok=True)
        for rel in required_paths:
            src = repo_root / rel
            dst = tmp_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        env = dict(**os.environ)
        env["PYTHONPATH"] = "src"
        subprocess.run(
            ["python", "scripts/predict.py", "--output-path", "predictions.csv"],
            cwd=tmp_root,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                "python",
                "scripts/validate_submission.py",
                "--predictions-path",
                "predictions.csv",
                "--test-path",
                "data/merged/test_full.csv",
            ],
            cwd=tmp_root,
            env=env,
            check=True,
        )
        shutil.copy2(tmp_root / "predictions.csv", repo_root / "artifacts" / "predictions_clean_run.csv")


if __name__ == "__main__":
    run()
    print("Clean dry-run completed successfully.")

