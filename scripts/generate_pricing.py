import logging
from pathlib import Path

from product_bundle.dataset import read_dataset
from product_bundle.utils import save_pricing_data


def main():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    data_dir = Path("data")
    df = read_dataset(data_dir).loc[:, ["StockCode", "UnitPrice", "InvoiceDate"]]
    logging.debug("Saving pricing data")
    save_pricing_data(df)
    logging.info("DONE")


if __name__ == "__main__":
    main()
