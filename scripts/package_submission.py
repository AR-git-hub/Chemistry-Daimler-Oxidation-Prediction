"""Create final submission zip with required files."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def create_submission_zip(predictions: Path, notebook: Path, output_zip: Path) -> Path:
    if not predictions.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions}")
    if not notebook.exists():
        raise FileNotFoundError(f"Missing notebook file: {notebook}")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zf:
        zf.write(predictions, arcname="predictions.csv")
        zf.write(notebook, arcname="inference.ipynb")
    return output_zip


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package required submission files into zip.")
    parser.add_argument("--predictions-path", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--notebook-path", type=Path, default=Path("notebooks/inference.ipynb"))
    parser.add_argument("--output-zip", type=Path, default=Path("artifacts/submission.zip"))
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    out = create_submission_zip(args.predictions_path, args.notebook_path, args.output_zip)
    print(f"Created submission archive: {out}")

