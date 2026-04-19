from pathlib import Path

import pandas as pd

RYABKOV_TABLE = "test_predictions.csv"
VLAD_TABLE = "test_predictions_9.csv"
RESULT_TABLE = "1.csv"

def merge_for_model():
    train = pd.read_csv(RYABKOV_TABLE)
    preds = pd.read_csv(VLAD_TABLE)

    targets = [
        "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
        "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"
    ]

    # scenario_id -> новое значение targets[1]
    mapping = preds.set_index("scenario_id")[targets[1]]
    train = train.drop(columns=[targets[1]])

    # заменить только нужную колонку
    train.insert(2, targets[1], train["scenario_id"].map(mapping))
    train = train.sort_values(
        "scenario_id",
        key=lambda s: s.str.split('_').str[1].astype(int)
    )

    train.to_csv(RESULT_TABLE, index=False, decimal=".")


if __name__ == "__main__":
    merge_for_model()