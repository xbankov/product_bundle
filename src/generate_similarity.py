import logging
from pathlib import Path


from dataset import read_dataset, text_and_price_preprocess


def main():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    data_dir = Path("data/data.csv")
    df = read_dataset(data_dir, force=False).loc[
        :, ["ItemID", "Description", "UnitPrice"]
    ]
    df = text_and_price_preprocess(df)
    


if __name__ == "__main__":
    main()
