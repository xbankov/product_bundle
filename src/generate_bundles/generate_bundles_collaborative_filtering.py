import logging
from pathlib import Path


from dataset import read_dataset
from utils import save_similar_products
from bundling_association_rules import add_text_and_price_features, get_similar_products


def main():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    data_dir = Path("data/data.csv")
    df = read_dataset(data_dir, force=False).loc[
        :, ["ItemID", "Description", "UnitPrice", "InvoiceDate"]
    ]
    text_df = add_text_and_price_features(df)
    similar_products = get_similar_products(text_df, num_products=3)
    save_similar_products(similar_products)
    logging.info("DONE")


if __name__ == "__main__":
    main()
