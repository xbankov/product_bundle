import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer


def content_filtering_product_bundle(
    df: pd.DataFrame, bundle_size: int = 3, max_distance: float = 0.5
):
    temp_path = Path("../data/features.csv")

    if temp_path.exists():
        df = pd.read_csv(temp_path)
    else:
        df = encode_description(df)
        df = encode_unit_price(df)
        df.to_csv(temp_path, index=False)

    features = df.iloc[:, 8:]

    knn = NearestNeighbors(n_neighbors=bundle_size, metric="cosine")

    knn.fit(features)

    neighbor_scores, neighbor_indices = knn.kneighbors(
        features, n_neighbors=bundle_size + 1
    )
    neighbor_indices = pd.DataFrame(neighbor_indices, index=df.loc[:, "ItemID"])

    for i in range(0, 4):
        neighbor_indices.iloc[:, i] = neighbor_indices.iloc[:, i].map(
            dict(zip(list(range(len(df))), df.loc[:, "ItemID"]))
        )

    neighbor_scores = pd.DataFrame(neighbor_scores, index=df.loc[:, "ItemID"])

    bundles = set()
    for i in range(len(df)):
        bundle = set()
        for score, idx in zip(
            neighbor_scores.iloc[i, 1:], neighbor_indices.iloc[i, 1:]
        ):
            if score < max_distance:
                bundle.add(idx)
        bundles.add(frozenset(bundle))

    return frozenset({bundle for bundle in bundles if len(bundle) > 1})


def encode_description(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Encoding description embeddings")
    df["Description"] = df["Description"].str.lower()
    df["Description"] = df["Description"].str.replace(r"[^\w\s]", "")

    logging.debug("Load the DistilBERT tokenizer and model")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    df = df.groupby(["Description", "ItemID"], as_index=False).first()

    tqdm.pandas()
    logging.info("Encoding textual information")
    df["EncodedText"] = df["Description"].progress_apply(
        partial(encode_text, tokenizer=tokenizer, model=model)
    )

    logging.debug("Unravelling encoded text and one-hot encoding the price")
    df_unraveled = df["EncodedText"].apply(pd.Series)

    df_unraveled.columns = [f"Dimension_{i+1}" for i in range(df_unraveled.shape[1])]
    df = pd.concat([df, df_unraveled], axis=1)
    df = df.drop(columns=["EncodedText"], axis=1)
    return df


def encode_unit_price(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Encoding unitprice")
    df.sort_values("InvoiceDate", ascending=False).groupby(
        ["Description", "ItemID"], as_index=False
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
