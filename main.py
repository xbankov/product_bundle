import logging
from pathlib import Path

from dataset import read_dataset, split
from bundling import rule_based_product_bundle
from evaluation import evaluate_bundles


def main():
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.info("Load and preprocess dataset (load preprocessed file if possible)")
    dataset = read_dataset(Path("data/data.csv"), force=True)

    logging.info("Split Dataset (load splits files if possible)")
    train_df, valid_df, test_df = split(dataset, "CustomerID")

    logging.info("Compute rule-based methods")

    # min_support between 0.01 and 0.1
    # min_threshold (lift) ==1.0
    # min_bundle_size 2
    # min_confidence between 0.5 and 0.8,
    # best_params = {
    #     "min_support": None,
    #     "min_bundle_size": None,
    #     "min_confidence": None,
    #     "precision": None,
    #     "recall": None,
    #     "f1": None,
    # }

    bundles = rule_based_product_bundle(
        train_df,
        min_support=0.01,
        metric="confidence",
        min_threshold=0.8,
        min_confidence=0.8,
        min_bundle_size=1,
    )

    precision, recall, f1 = evaluate_bundles(valid_df, bundles)
    print(precision, recall, f1)
    # Add features

    # Train ML methods

    # Evaluate

    # Save the best one and serve it in REST API in Docker


if __name__ == "__main__":
    main()
