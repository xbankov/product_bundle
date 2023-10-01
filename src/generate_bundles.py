import logging
from pathlib import Path
from tqdm import tqdm

# import wandb
from sklearn.model_selection import ParameterGrid
from dataset import read_dataset, split
from evaluation import evaluate_bundles
from bundling import rule_based_product_bundle
from utils import save_best_bundles


def main():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    data_dir = Path("data/data.csv")
    logging.info("Load and preprocess dataset (load preprocessed file if possible)")
    dataset = read_dataset(data_dir, force=False)

    logging.info("Split Dataset (load splits files if possible)")
    train_df, valid_df, test_df = split(dataset, data_dir, "InvoiceNo", force=False)

    logging.info("Grid-search over rule-based apriori hyper-parameters")
    # Define the hyperparameter grid for grid search
    param_grid = {
        "min_support": [0.02],
        "min_confidence": [0.75],
        "min_bundle_size": [1],
        "metric": ["confidence"],
        "min_threshold": [0.75],
    }
    # Initialize Weights and Biases (wandb)
    # wandb.init(project="product-bundle-hyperparameter-search")

    # Create a dictionary to store the best hyperparameters and evaluation scores
    best_params = {
        "min_support": None,
        "min_confidence": None,
        "min_bundle_size": None,
        "metric": None,
        "min_threshold": None,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    # Iterate through the hyperparameter grid
    for params in tqdm(ParameterGrid(param_grid)):
        # wandb.config.update(
        # params, allow_val_change=True
        # )  # Log hyperparameters to wandb

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

        # Log evaluation metrics to wandb
        # wandb.log({"precision": precision, "recall": recall, "f1": f1})

        # Check if the current hyperparameters outperform the best ones found so far
        if f1 > best_params["f1"]:
            best_params["min_support"] = params["min_support"]
            best_params["min_confidence"] = params["min_confidence"]
            best_params["min_bundle_size"] = params["min_bundle_size"]
            best_params["precision"] = precision
            best_params["recall"] = recall
            best_params["f1"] = f1
            save_best_bundles(bundles)

    # Print the best hyperparameters and their corresponding evaluation scores
    print("Best Hyperparameters:")
    print(best_params)

    # Run the rule-based product bundling with the best hyperparameters on the test dataset
    bundles = rule_based_product_bundle(
        train_df,
        min_support=best_params["min_support"],
        metric="confidence",
        min_threshold=0.8,
        min_confidence=best_params["min_confidence"],
        min_bundle_size=best_params["min_bundle_size"],
    )

    # Evaluate the bundles on the test dataset
    precision, recall, f1 = evaluate_bundles(test_df, bundles)

    # Print the evaluation metrics on the test dataset
    print("Evaluation Metrics on Test Data:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


if __name__ == "__main__":
    main()
