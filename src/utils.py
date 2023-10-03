import pickle
from typing import Dict, FrozenSet, List

import pandas as pd


def save_best_params(params: Dict) -> None:
    with open("static/best_params.pickle", "wb") as pickle_file:
        pickle.dump(params, pickle_file)


def load_best_params() -> Dict:
    with open("static/best_params.pickle", "rb") as pickle_file:
        return pickle.load(pickle_file)


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


def save_similar_products(df: pd.DataFrame):
    return df.to_csv("static/item_similarity.csv", index=False)


def load_similar_products() -> pd.DataFrame:
    return pd.read_csv("static/item_similarity.csv")
