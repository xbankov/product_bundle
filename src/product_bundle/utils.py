import logging
import pickle
from functools import partial
from pathlib import Path
from typing import Dict, FrozenSet, Set, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer


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
    df.to_csv("static/pricing.csv", index=False)


def load_pricing_data() -> pd.DataFrame:
    return pd.read_csv("static/pricing.csv")


def load_all_bundles() -> (
    Tuple[Set[FrozenSet[str]], Set[FrozenSet[str]], Set[FrozenSet[str]]]
):
    rule_bundles = load_bundles("rule")
    collaborative_bundles = load_bundles("collaborative")
    content_bundles = load_bundles("content")
    return rule_bundles, collaborative_bundles, content_bundles


def load_product_df(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "product_df.csv")


def prepare_product_df(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    product_df_path = data_dir / "/product_df.csv"

    if not product_df_path.exists():
        product_df = (
            df.sort_values(by="InvoiceDate", ascending=False)
            .groupby(["StockCode"], as_index=False)
            .first()
        )
        product_df = prepare_text_embeddings(product_df)
        product_df = prepare_encoded_unit_price(product_df)
        product_df.to_csv(product_df_path, index=False)

    return pd.read_csv(product_df_path)


def prepare_text_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Encoding description embeddings")
    df["Description"] = df["Description"].str.lower()
    df["Description"] = df["Description"].str.replace(r"[^\w\s]", "")

    logging.debug("Load the DistilBERT tokenizer and model")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    df = df.groupby(["Description", "StockCode"], as_index=False).first()

    tqdm.pandas()
    logging.info("Encoding textual information")
    df["EncodedText"] = df["Description"].progress_apply(
        partial(encode_text, tokenizer=tokenizer, model=model)
    )

    logging.debug("Unravelling encoded text")
    df_unraveled = df["EncodedText"].apply(pd.Series)

    df_unraveled.columns = [f"Dimension_{i+1}" for i in range(df_unraveled.shape[1])]
    df = pd.concat([df, df_unraveled], axis=1)
    df = df.drop(columns=["EncodedText"], axis=1)
    return df


def prepare_encoded_unit_price(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Encoding unitprice")
    df.sort_values("InvoiceDate", ascending=False).groupby(
        ["Description", "StockCode"], as_index=False
    )

    conditions = [
        df["UnitPrice"] < 1.0,
        df["UnitPrice"].between(1.0, 2.0),
        df["UnitPrice"].between(2.0, 5.0),
        df["UnitPrice"] >= 5.0,
    ]

    labels = ["bin1", "bin2", "bin3", "bin4"]

    one_hot_encoded = pd.get_dummies(
        pd.DataFrame(conditions, index=labels).T, prefix="", prefix_sep=""
    )

    df = pd.concat([df, one_hot_encoded], axis=1)
    return df


def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
