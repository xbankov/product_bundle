import logging
from typing import FrozenSet, List
import pandas as pd
import pickle
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def rule_based_product_bundle(
    df,
    min_support=0.01,
    min_threshold=1.0,
    metric="lift",
    min_confidence=0.5,
    min_bundle_size=1,
):
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


def save_best_bundles(bundles: List[FrozenSet]) -> None:
    with open("static/best_bundles.pickle", "wb") as pickle_file:
        pickle.dump(bundles, pickle_file)


def load_best_bundles() -> List[FrozenSet]:
    with open("static/best_bundles.pickle", "rb") as pickle_file:
        return pickle.load(pickle_file)
