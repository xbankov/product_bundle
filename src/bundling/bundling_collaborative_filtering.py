from sklearn.neighbors import NearestNeighbors
import pandas as pd


def collaborative_filtering_product_bundle(
    df: pd.DataFrame, bundle_size: int = 3, max_distance: float = 0.5, **kwargs
):
    basket = pd.pivot_table(
        df,
        index="ItemID",
        columns="CustomerID",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    ).astype(bool)

    knn = NearestNeighbors(n_neighbors=bundle_size, metric="cosine")
    knn.fit(basket)

    neighbor_scores, neighbor_indices = knn.kneighbors(
        basket, n_neighbors=bundle_size + 1
    )
    neighbor_indices = pd.DataFrame(neighbor_indices, index=basket.index)

    for i in range(0, 4):
        neighbor_indices.iloc[:, i] = neighbor_indices.iloc[:, i].map(
            dict(zip(list(range(len(basket))), basket.index))
        )

    neighbor_scores = pd.DataFrame(neighbor_scores, index=basket.index)

    bundles = set()
    for i in range(len(basket)):
        bundle = set()
        for score, idx in zip(
            neighbor_scores.iloc[i, 1:], neighbor_indices.iloc[i, 1:]
        ):
            if score < max_distance:
                bundle.add(idx)
        bundles.add(frozenset(bundle))

    return frozenset({bundle for bundle in bundles if len(bundle) > 1})
