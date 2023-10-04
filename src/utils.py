from ast import Set
import pickle
from typing import Dict, FrozenSet, List, Tuple

import pandas as pd


def save_best_params(method: str, params: Dict) -> None:
    with open(f"static/best_params_{method}.pickle", "wb") as pickle_file:
        pickle.dump(params, pickle_file)


def load_best_params(method: str) -> Dict:
    with open(f"static/best_params_{method}.pickle", "rb") as pickle_file:
        return pickle.load(pickle_file)


def save_bundles(method: str, bundles: Set[FrozenSet[str]]) -> None:
    with open(f"static/{method}_bundles.pickle", "wb") as pickle_file:
        pickle.dump(bundles, pickle_file)


def load_bundles(method: str) -> Set[FrozenSet[str]]:
    with open(f"static/{method}_bundles.pickle", "rb") as pickle_file:
        return pickle.load(pickle_file)


def save_pricing_data(df: pd.DataFrame):
    return df.to_csv("static/pricing.csv", index=False)


def load_pricing_data() -> pd.DataFrame:
    return pd.read_csv("static/pricing.csv")


def load_all_bundles() -> (
    Tuple[Set[FrozenSet[str]], Set[FrozenSet[str]], Set[FrozenSet[str]]]
):
    rule_bundles = load_bundles("rule")
    collaborative_bundles = load_bundles("collaborative")
    content_bundles = load_bundles("content")
    return rule_bundles, collaborative_bundles, content_bundles
