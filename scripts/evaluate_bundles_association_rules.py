import logging
from pathlib import Path

from product_bundle.dataset import preprocess, read_dataset, split
from product_bundle.evaluation import evaluation_loop


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    data_dir = Path("data")

    logging.debug("Load and preprocess dataset (load preprocessed file if possible)")
    dataset = read_dataset(data_dir)

    logging.debug("Split Dataset (load splits files if possible)")
    train_df, valid_df, test_df = split(dataset, data_dir, "InvoiceNo", force=False)

    param_grid = {
        "min_support": [0.02, 0.01],
        "min_confidence": [0.5, 0.75, 0.9],
        "min_bundle_size": [1, 2],
        "metric": ["confidence", "lift"],
        "min_threshold": [0.5, 0.75, 0.9, 1.0],
    }

    evaluation_loop(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        method="rule",
        param_grid=param_grid,
    )


if __name__ == "__main__":
    main()
