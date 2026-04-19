"""Central constants for the DOT task."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "merged"
ARTIFACTS_DIR = ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "train_full.csv"
TEST_PATH = DATA_DIR / "test_full.csv"

# Очищенная таблица признаков (те же scenario_id / таргеты; другой набор колонок).
TRAIN_FE_CLEANED_PATH = DATA_DIR / "train_fe_cleaned.csv"
TEST_FE_CLEANED_PATH = DATA_DIR / "test_fe_cleaned.csv"

# Расширенная таблица признаков (COC, CCS -20..-35, tbn_consolidated, ...). Строки 1:1 с train_full.
TRAIN_FE_PATH = DATA_DIR / "train_fe.csv"
TEST_FE_PATH = DATA_DIR / "test_fe.csv"

# train_full + колонка tbn_consolidated из train_fe_cleaned (см. scripts/merge_tbn_from_fe_cleaned.py).
TRAIN_FULL_TBN_PATH = DATA_DIR / "train_full_plus_tbn.csv"
TEST_FULL_TBN_PATH = DATA_DIR / "test_full_plus_tbn.csv"

TARGET_COLUMNS = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
]

ID_COLUMNS = ["scenario_id", "Компонент", "Наименование партии"]
SCENARIO_ID = "scenario_id"

# Extra numeric columns to drop (IDs/targets already removed in ``_feature_columns``).
# Party name is an identifier / batch surrogate; viscosity at -30°C is constant in ``train_full``.
FEATURE_BLOCKLIST: frozenset[str] = frozenset(
    {
        "Наименование партии",
        "Кинематическая вязкость, при -30⁰С | ASTM D445",
    }
)

