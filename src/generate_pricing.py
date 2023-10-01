import logging
from pathlib import Path

from dataset import read_dataset
from utils import save_pricing_data


def main():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    data_dir = Path("data/data.csv")
    df = read_dataset(data_dir, force=False).loc[
        :, ["ItemID", "UnitPrice", "InvoiceDate"]
    ]
    logging.debug("Saving pricing data")
    save_pricing_data(df)


if __name__ == "__main__":
    main()
