import logging
from pathlib import Path

from dataset import read_dataset, split
from bundling import rule_based_product_bundle
from evaluation import evaluate_bundles

from fastapi import FastAPI

app = FastAPI()


with open("settings.pickle", "rb") as file:
    bundles = pickle.load(file)


@app.get("/bundles/{product_id}")
def get_product_bundle(product_id: str):
    # Logic to retrieve the product bundle based on the product_id
    # Calculate the price for the whole bundle
    # Return the list of products and the price for the bundle

    return {"product_id": product_id, "products": [...], "price": ...}
