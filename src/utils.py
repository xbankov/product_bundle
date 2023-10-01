import pickle
from typing import FrozenSet, List

import pandas as pd


def save_best_bundles(bundles: List[FrozenSet]) -> None:
    with open("static/best_bundles.pickle", "wb") as pickle_file:
        pickle.dump(bundles, pickle_file)


def load_best_bundles() -> List[FrozenSet]:
    with open("static/best_bundles.pickle", "rb") as pickle_file:
        return pickle.load(pickle_file)


def save_pricing_data(df: pd.DataFrame):
    return df.to_csv("static/pricing.csv", index=False)


def load_pricing_data() -> pd.DataFrame:
    return pd.read_csv("static/pricing.csv")


def save_similarity_matrix(df: pd.DataFrame):
    return df.to_csv("static/item_similarity.csv", index=False)


def load_similarity_matrix() -> pd.DataFrame:
    return pd.read_csv("static/item_similarity.csv")
