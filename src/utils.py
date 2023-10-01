from pathlib import Path
import pickle
from typing import FrozenSet, List

import pandas as pd

from src.dataset import read_dataset


def save_best_bundles(bundles: List[FrozenSet]) -> None:
    with open("static/best_bundles.pickle", "wb") as pickle_file:
        pickle.dump(bundles, pickle_file)


def load_best_bundles() -> List[FrozenSet]:
    with open("static/best_bundles.pickle", "rb") as pickle_file:
        return pickle.load(pickle_file)


def save_pricing_data(df: pd.DataFrame):
    return pd.to_csv(df, "static/pricing.csv", index=False)


def load_pricing_data() -> pd.DataFrame:
    return pd.read_csv("static/pricing.csv")


def generate_pricing_data():
    data_dir = Path("data/data.csv")
    dataset = read_dataset(data_dir, force=False).loc[:, ["ItemID", "UnitPrice"]]
    save_pricing_data(dataset)
