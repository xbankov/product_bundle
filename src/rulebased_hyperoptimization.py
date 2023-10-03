import logging
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import ParameterGrid
from dataset import read_dataset, split
from evaluation import evaluate_bundles
from bundling import rule_based_product_bundle
from utils import save_best_bundles, save_best_params


def main():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    data_dir = Path("data/data.csv")
    logging.debug("Load and preprocess dataset (load preprocessed file if possible)")
    dataset = read_dataset(data_dir, force=False)

    logging.debug("Split Dataset (load splits files if possible)")
    train_df, valid_df, test_df = split(dataset, data_dir, "InvoiceNo", force=False)

    logging.debug("Grid-search over rule-based apriori hyper-parameters")
    # Define the hyperparameter grid for grid search
    param_grid = {
        "min_support": [0.02, 0.1],
        "min_confidence": [0.5, 0.75, 0.9],
        "min_bundle_size": [1, 2],
        "metric": ["confidence", "lift"],
        "min_threshold": [0.5, 0.75, 0.9, 1.0],
    }

    # Create a dictionary to store the best hyperparameters and evaluation scores
    best_params = {
        "min_support": [None],
        "min_confidence": None,
        "min_bundle_size": None,
        "metric": None,
        "min_threshold": None,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    logging.info("Iterate through the hyperparameter grid")
    for params in tqdm(ParameterGrid(param_grid)):
        # Run the rule-based product bundling with the current hyperparameters
        bundles = rule_based_product_bundle(
            train_df,
            min_support=params["min_support"],
            metric=params["metric"],
            min_threshold=params["min_threshold"],
            min_confidence=params["min_confidence"],
            min_bundle_size=params["min_bundle_size"],
        )

        try:
            # Evaluate the bundles on the validation dataset
            precision, recall, f1 = evaluate_bundles(valid_df, bundles)
        except ZeroDivisionError:
            precision, recall, f1 = 0, 0, 0

        # Check if the current hyperparameters outperform the best ones found so far
        if f1 > best_params["f1"]:
            best_params["min_support"] = params["min_support"]
            best_params["min_confidence"] = params["min_confidence"]
            best_params["min_bundle_size"] = params["min_bundle_size"]
            best_params["metric"] = params["metric"]
            best_params["min_threshold"] = params["min_threshold"]

            best_params["precision"] = precision
            best_params["recall"] = recall
            best_params["f1"] = f1
            save_best_bundles(bundles)

    # Print the best hyperparameters and their corresponding evaluation scores
    logging.info("Best Hyperparameters:")
    logging.info(best_params)

    save_best_params(best_params)

    # Evaluate the bundles on the test dataset
    try:
        # Evaluate the bundles on the validation dataset
        precision, recall, f1 = evaluate_bundles(test_df, bundles)
    except ZeroDivisionError:
        logging.warning("No bundles for test_df were found :( :(")
        precision, recall, f1 = 0, 0, 0

    # Print the evaluation metrics on the test dataset
    logging.info("Evaluation Metrics on Test Data:")
    logging.info("Precision:", precision)
    logging.info("Recall:", recall)
    logging.info("F1 Score:", f1)


if __name__ == "__main__":
    main()
