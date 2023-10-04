import logging
from pathlib import Path

from dataset import read_dataset, split
from evaluation import evaluate_bundles, evaluation_loop
from utils import load_all_bundles


def main():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    data_dir = Path("data/data.csv")
    logging.debug("Load and preprocess dataset (load preprocessed file if possible)")
    dataset = read_dataset(data_dir, force=False)

    logging.debug("Split Dataset (load splits files if possible)")
    _, _, test_df = split(dataset, data_dir, "InvoiceNo", force=True)

    rule_bundles, collaborative_bundles, content_bundles = load_all_bundles()
    bundles = rule_bundles | collaborative_bundles | content_bundles

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


if __name__ == "__main__":
    main()
