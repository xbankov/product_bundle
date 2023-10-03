import logging
from functools import partial
from typing import FrozenSet, List

import numpy as np
import pandas as pd
import torch
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer


def rule_based_product_bundle(
    df: pd.DataFrame,
    min_support: float = 0.01,
    min_threshold: float = 1.0,
    metric: str = "lift",
    min_confidence: float = 0.5,
    min_bundle_size: int = 1,
) -> List[FrozenSet]:
    logging.debug(
        "Create a pivot table for market basket analysis using InvoiceNo and ItemID"
    )
    basket = pd.pivot_table(
        df,
        index="InvoiceNo",
        columns="ItemID",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    ).astype(bool)

    logging.debug("Perform market basket analysis using Apriori algorithm")
    frequent_itemsets = apriori(
        basket, min_support=min_support, use_colnames=True, low_memory=True
    )

    common_bundles = {frozenset(s) for s in frequent_itemsets["itemsets"] if len(s) > 1}
    logging.debug(
        f"Extracted {len(common_bundles)} common_bundles from frequent itemsets"
    )

    logging.debug("Generate association rules from frequent itemsets")
    rules = association_rules(
        frequent_itemsets,
        metric=metric,
        min_threshold=min_threshold,
    )

    logging.debug("Sort the rules by confidence or lift, depending on your preference")
    sorted_rules = rules.sort_values(by=["confidence"], ascending=False)

    generated_bundles = set()

    logging.debug("Iterate through the sorted rules and extract bundles")
    for _, rule in sorted_rules.iterrows():
        antecedents = list(rule["antecedents"])
        consequents = list(rule["consequents"])

        # Check if the rule meets the criteria
        if rule["confidence"] >= min_confidence and len(consequents) >= min_bundle_size:
            bundle = antecedents + consequents
            generated_bundles.add(frozenset(bundle))
    logging.debug(f"Generated {len(generated_bundles)} bundles using association rules")
    return common_bundles | generated_bundles


def add_text_and_price_features(df: pd.DataFrame) -> pd.DataFrame:
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
    logging.info("Encoding textual information")
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
    df = df.drop(columns=["EncodedText"], axis=1)
    return df


def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def get_similar_products(text_df, num_products=3):
    similarity_matrix = cosine_similarity(text_df.iloc[:, 3:], text_df.iloc[:, 3:])
    sorted_similarities_indixes = np.argsort(similarity_matrix, axis=0)[::-1][
        1 : num_products + 1
    ].transpose(1, 0)

    sorted_similarities = np.sort(similarity_matrix, axis=0)[::-1][
        1 : num_products + 1
    ].transpose(1, 0)
    similar_products = pd.concat(
        [
            text_df.iloc[:, :3],
            pd.DataFrame(
                sorted_similarities_indixes,
                columns=["Top Index", "Second Index", "Third Index"],
            ),
            pd.DataFrame(
                sorted_similarities,
                columns=["Top Score", "Second Score", "Third Score"],
            ),
        ],
        axis=1,
    )
    return similar_products
