from pathlib import Path

import pandas as pd

OUTPUT = Path(__file__).resolve().parent / "data/output"
TABLE_TO_PREPARE = "test_predictions.csv"
RESULT_TABLE = "./res.csv"

def merge_for_model():
    train = pd.read_csv(OUTPUT / "test/test_full.csv")
    preds = pd.read_csv(TABLE_TO_PREPARE)

    targets = [
        "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
        "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"
    ]

    # scenario_id -> новое значение targets[0]
    mapping = preds.set_index("scenario_id")[targets[0]]

    # заменить только нужную колонку
    train.insert(6, targets[0], train["scenario_id"].map(mapping))

    train.to_csv(RESULT_TABLE, index=False, decimal=".")


if __name__ == "__main__":
    merge_for_model()