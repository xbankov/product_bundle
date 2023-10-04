import logging
from pathlib import Path
from typing import Dict, List, FrozenSet
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from bundling_association_rules import rule_based_product_bundle
from bundling_collaborative_filtering import collaborative_filtering_product_bundle
from bundling_content_based_filtering import content_filtering_product_bundle

from dataset import read_dataset, split
from utils import save_best_params, save_bundles


def evaluate_bundles(df: pd.DataFrame, predicted_bundles: Set[FrozenSet], verbose=0):
    # Convert the testing DataFrame into a set of actual bundles
    setty_df = df.groupby("InvoiceNo")["ItemID"].apply(frozenset)
    actual_bundles = set(setty_df[setty_df.apply(len) > 1].to_list())

    # Calculate evaluation metrics
    true_positives = len(predicted_bundles & actual_bundles)
    false_positives = len(predicted_bundles - actual_bundles)
    false_negatives = len(actual_bundles - predicted_bundles)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    if verbose:
        print("Evaluation Metrics:")
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)

    return precision, recall, f1_score


def evaluation_loop(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str,
    param_grid: Dict[str, List],
):
    logging.debug(f"Grid-search over {method}-based parameter_grid {param_grid}")

    # Create a dictionary to store the best hyperparameters and evaluation scores
    best_params = {
        **param_grid,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    logging.info("Iterate through the hyperparameter grid")
    for params in tqdm(ParameterGrid(param_grid)):
        logging.debug(f"Run the {method}-based product bundling")
        if method == "rule":
            bundles = rule_based_product_bundle(train_df, **params)
        elif method == "collaborative":
            bundles = collaborative_filtering_product_bundle(train_df, **params)
        elif method == "content":
            bundles = content_filtering_product_bundle(train_df, **params)
        else:
            raise NotImplementedError(
                f"{method} not supported. Please choose one of: ['rule', 'collaborative', 'content'] "
            )

        try:
            # Evaluate the bundles on the validation dataset
            precision, recall, f1 = evaluate_bundles(valid_df, bundles)
        except ZeroDivisionError:
            precision, recall, f1 = 0, 0, 0

        # Check if the current hyperparameters outperform the best ones found so far
        if precision >= best_params["precision"]:
            best_params.update(params)

            best_params["precision"] = precision
            best_params["recall"] = recall
            best_params["f1"] = f1

            save_bundles(method=method, bundles=bundles)

    # Print the best hyperparameters and their corresponding evaluation scores
    logging.info("Best Hyperparameters:")
    logging.info(best_params)

    save_best_params(method=method, params=best_params)

    logging.debug("Evaluate the bundles on the test dataset")
    try:
        precision, recall, f1 = evaluate_bundles(test_df, bundles)
    except ZeroDivisionError:
        logging.warning("No bundles for test_df were found :( :(")
        precision, recall, f1 = 0, 0, 0

    # Print the evaluation metrics on the test dataset
    logging.info("Evaluation Metrics on test_df:")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")
