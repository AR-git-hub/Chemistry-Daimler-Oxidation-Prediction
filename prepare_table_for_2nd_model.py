from pathlib import Path

import pandas as pd

OUTPUT = Path(__file__).resolve().parent / "data/output"
TABLE_TO_PREPARE = "test_predictions.csv"

def merge_for_model():
    train = pd.read_csv(OUTPUT / "train/train_full.csv")
    preds = pd.read_csv(TABLE_TO_PREPARE)
    targets = ["Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %", "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"]
    res = train.merge(preds, on=targets[0])
    res.to_csv("res.csv", index=False, decimal=".")


if __name__ == "__main__":
    merge_for_model()