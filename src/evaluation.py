def evaluate_bundles(df, predicted_bundles, verbose=0):
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
