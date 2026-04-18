"""Central constants for the DOT task."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "merged"
ARTIFACTS_DIR = ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "train_full.csv"
TEST_PATH = DATA_DIR / "test_full.csv"

TARGET_COLUMNS = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
]

ID_COLUMNS = ["scenario_id", "Компонент", "Наименование партии"]
SCENARIO_ID = "scenario_id"

