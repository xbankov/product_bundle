from typing import FrozenSet, Set

import pandas as pd
import torch
import torch.nn as nn


def price_item(product_df: pd.DataFrame, product_id: str):
    return (
        product_df[product_df["StockCode"] == product_id]
        .sort_values("InvoiceDate", ascending=False)
        .iloc[0]["UnitPrice"]
    )


def price_bundle(
    product_df: pd.DataFrame, bundle: Set[FrozenSet], discount: float = 1.0
):
    return discount * sum(price_item(product_df, product_id) for product_id in bundle)


class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
