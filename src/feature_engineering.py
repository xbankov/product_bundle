import logging
from functools import partial

import pandas as pd
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer

from bundling import encode_text


def text_and_price(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Encoding description embeddings")
    df["Description"] = df["Description"].str.lower()
    df["Description"] = df["Description"].str.replace(r"[^\w\s]", "")

    logging.debug("Load the DistilBERT tokenizer and model")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    df = (
        df.sort_values("InvoiceDate", ascending=False)
        .groupby(["Description", "ItemID"], as_index=False)
        .first()[["Description", "ItemID", "UnitPrice"]]
    )

    tqdm.pandas()
    df["EncodedText"] = df["Description"].progress_apply(
        partial(encode_text, tokenizer=tokenizer, model=model)
    )

    logging.debug("Unravelling encoded text and one-hot encoding the price")
    df_unraveled = df["EncodedText"].apply(pd.Series)

    # Rename the new columns if needed
    df_unraveled.columns = [f"Dimension_{i+1}" for i in range(df_unraveled.shape[1])]

    # Define custom conditions to create bins
    conditions = [
        df["UnitPrice"] < 1.0,
        df["UnitPrice"].between(1.0, 2.0),
        df["UnitPrice"].between(2.0, 5.0),
        df["UnitPrice"] >= 5.0,
    ]

    # Create a list of labels for the bins
    labels = ["bin1", "bin2", "bin3", "bin4"]

    # Use pd.get_dummies to one-hot encode based on conditions
    one_hot_encoded = pd.get_dummies(
        pd.DataFrame(conditions, index=labels).T, prefix="", prefix_sep=""
    )

    df = pd.concat([df, df_unraveled, one_hot_encoded], axis=1)
    df = df.drop(columns=["EncodedText", "Description", "UnitPrice"], axis=1)
