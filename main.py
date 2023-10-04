from fastapi import FastAPI
from pricing import price_bundle

from src.utils import load_pricing_data, load_bundles


app = FastAPI()

rule_bundles, collaborative_bundles, content_bundles = load_bundles()
pricing_dataset = load_pricing_data()


@app.get("/bundles/{product_id}")
def get_product_bundle(product_id: str):
    product_bundle_rule = [bundle for bundle in rule_bundles if product_id in bundle]
    product_bundle_collaborative = [
        bundle for bundle in collaborative_bundles if product_id in bundle
    ]
    product_bundle_content = [
        bundle for bundle in content_bundles if product_id in bundle
    ]

    if len(product_bundle_rule) > 0:
        sorted_bundles = sorted(product_bundle_rule, key=len, reverse=True)
        bundle = sorted_bundles[0]
    elif len(product_bundle_collaborative) > 0:
        bundle = product_bundle_collaborative[0]

    elif len(product_bundle_content) > 0:
        bundle = product_bundle_collaborative[0]

    else:
        bundle = [product_id]

    # Calculate the price for the whole bundle
    price = price_bundle(pricing_dataset, bundle)

    # Return the list of products and the price for the bundle
    return {"product_id": product_id, "products": bundle, "price": price}
