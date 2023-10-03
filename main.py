from pathlib import Path
from fastapi import FastAPI
from src.pricing import price_bundle

from src.utils import load_best_bundles, load_pricing_data, load_similar_products


app = FastAPI()


static_bundles = load_best_bundles()
pricing_dataset = load_pricing_data()
similar_products = load_similar_products()


@app.get("/bundles/{product_id}")
def get_product_bundle(product_id: str):
    # Logic to retrieve the product bundle based on the product_id
    # Filter the frozensets that contain the productId
    filtered_bundles = [bundle for bundle in static_bundles if product_id in bundle]

    # Sort the filtered bundles by length in descending order (Ideally pick by metric)
    sorted_bundles = sorted(filtered_bundles, key=len, reverse=True)

    if sorted_bundles:
        bundle = sorted_bundles[0]
    else:
        product_row = similar_products[similar_products["ItemID"] == product_id]
        bundle = product_row.iloc[["Top Index", "Second Index", "Third Index"]][
            "ItemID"
        ]

    # Calculate the price for the whole bundle
    price = price_bundle(pricing_dataset, bundle)

    # Return the list of products and the price for the bundle
    return {"product_id": product_id, "products": bundle, "price": price}
