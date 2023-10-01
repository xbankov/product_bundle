from typing import FrozenSet, List
import pandas as pd


def price_item(pricing_df: pd.DataFrame, product_id: str):
    return (
        pricing_df[pricing_df["ItemID"] == product_id]
        .sort_values("InvoiceDate", ascending=False)
        .iloc[0]["UnitPrice"]
    )


def price_bundle(bundle: List[FrozenSet], discount: float = 1.0):
    return discount * sum(price_item(product_id) for product_id in bundle)
