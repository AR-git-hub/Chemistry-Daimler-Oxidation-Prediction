"""Interaction feature notes and helpers."""

from __future__ import annotations

import json
from pathlib import Path

from dot.config import ARTIFACTS_DIR


def export_interaction_feature_manifest(metadata_path: Path | None = None) -> Path:
    """Persist names of interaction/context features used by the model."""
    metadata_path = metadata_path or (ARTIFACTS_DIR / "metadata.json")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    context_columns = metadata.get("context_columns", [])
    output = ARTIFACTS_DIR / "interaction_features.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump({"context_columns": context_columns}, f, ensure_ascii=False, indent=2)
    return output

