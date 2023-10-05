import pandas as pd
from sklearn.neighbors import NearestNeighbors

from product_bundle.utils import load_product_df


def content_filtering_product_bundle(
    df: pd.DataFrame, bundle_size: int = 3, max_distance: float = 0.5
):
    features = df.iloc[:, 8:]

    knn = NearestNeighbors(n_neighbors=bundle_size, metric="cosine")

    knn.fit(features)

    neighbor_scores, neighbor_indices = knn.kneighbors(
        features, n_neighbors=bundle_size + 1
    )
    neighbor_indices = pd.DataFrame(neighbor_indices, index=df.loc[:, "StockCode"])

    for i in range(0, bundle_size + 1):
        neighbor_indices.iloc[:, i] = neighbor_indices.iloc[:, i].map(
            dict(zip(list(range(len(df))), df.loc[:, "StockCode"]))
        )

    neighbor_scores = pd.DataFrame(neighbor_scores, index=df.loc[:, "StockCode"])

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
