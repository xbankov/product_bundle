import logging
from pathlib import Path
from tqdm import tqdm

from dataset import read_dataset
from bundling import rule_based_product_bundle
from utils import load_best_params, save_best_bundles


def main():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    data_dir = Path("data/data.csv")
    logging.debug("Load and preprocess dataset (load preprocessed file if possible)")
    dataset = read_dataset(data_dir, force=False)

    logging.info("Loading the best hyper-parameters")
    best_params = load_best_params()

    # Print the best hyperparameters and their corresponding evaluation scores
    print("Best Hyperparameters:")
    print(best_params)

    # Run the rule-based product bundling with the best hyperparameters on the test dataset
    bundles = rule_based_product_bundle(
        dataset,
        min_support=best_params["min_support"],
        metric=best_params["metric"],
        min_threshold=best_params["min_threshold"],
        min_confidence=best_params["min_confidence"],
        min_bundle_size=best_params["min_bundle_size"],
    )

    save_best_bundles(bundles)
    logging.info("DONE")


if __name__ == "__main__":
    main()
