import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset(path: Path, force: bool = False) -> pd.DataFrame:
    cleaned_path = path.parent / "data_cleaned.csv"
    if cleaned_path.exists() and not force:
        logging.debug(f"Loading preprocessed csv from {cleaned_path}")
        return pd.read_csv(cleaned_path)

    logging.debug(
        f"Cannot find preprocessed csv from {cleaned_path}. Preprocessing ..."
    )
    df_initial = pd.read_csv(
        path,
        encoding="ISO-8859-1",
        dtype={"CustomerID": str, "InvoiceID": str},
        parse_dates=["InvoiceDate"],
    )

    df_cleaned = preprocess(df_initial)
    df_cleaned.to_csv(cleaned_path, index=False)
    return pd.read_csv(cleaned_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Rename StockCode to ItemID")
    df = df.rename({"StockCode": "ItemID"}, axis=1)

    logging.debug("Remove rows with missing CustomerID")
    df = df[~df["CustomerID"].isna()]

    logging.debug("Duplicated rows: {}".format(df.duplicated().sum()))
    df = df.drop_duplicates()

    logging.debug("Resolving cancellations")
    df = resolve_cancellations(df)

    logging.debug("Removing special ItemID codes:")
    list_special_codes = df[df["ItemID"].str.contains("^[a-zA-Z]+", regex=True)][
        "ItemID"
    ].unique()

    logging.debug(f"{list_special_codes}")
    df = df[~df["ItemID"].isin(list_special_codes)]

    return df


def resolve_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Filter and select cancellations")
    cancellations = df[df["InvoiceNo"].str.startswith("C")]
    cancellations = cancellations.loc[
        :, ["ItemID", "CustomerID", "InvoiceDate", "Quantity"]
    ]

    logging.debug("Inner join positive quantity orders with cancellations")
    merged = pd.merge(
        df[df["Quantity"] > 0],
        cancellations,
        how="inner",
        on=["ItemID", "CustomerID"],
    )

    logging.debug("Cancellation has to be after the order")
    merged = merged[merged["InvoiceDate_x"] < merged["InvoiceDate_y"]]

    logging.debug("Remove duplicates")
    merged = merged.drop_duplicates()

    logging.debug("Substract cancellations until quantity would be negative")

    def resolve_cancel(df: pd.DataFrame) -> int:
        df = df.sort_values(by="InvoiceDate_y")
        quantity = df["Quantity_x"].iloc[0]
        for _, row in df.iterrows():
            if quantity + row["Quantity_y"] >= 0:
                quantity += row["Quantity_y"]
        return quantity

    logging.debug(
        "Group the cancellations per invoice and resolve quantity calculation"
    )
    merged = (
        merged.groupby(["InvoiceNo"])
        .apply(resolve_cancel)
        .reset_index(name="ResolvedQuantity")
    )

    logging.debug("Remove 0 as the product was cancelled completely")
    merged = merged[merged["ResolvedQuantity"] > 0]

    logging.debug("Merge back into transactions dataset")
    df_cleaned = pd.merge(df, merged, how="left", on="InvoiceNo")

    logging.debug(
        "Update Quantity with ResolvedQuantity, where available and drop the col"
    )
    mask_to_update = ~df_cleaned["ResolvedQuantity"].isna()
    df_cleaned.loc[mask_to_update, "Quantity"] = df_cleaned.loc[
        mask_to_update, "ResolvedQuantity"
    ]
    df_cleaned = df_cleaned.drop(["ResolvedQuantity"], axis=1)

    logging.debug("Remove all the cancellations, as they are resolved already")
    df_cleaned = df_cleaned[~df_cleaned["InvoiceNo"].str.startswith("C")]

    logging.debug("Reset the index")
    df_cleaned.reset_index(drop=True)

    return df_cleaned


def split(
    df: pd.DataFrame, data_dir: Path, col_name: str, force: bool = False
) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits_dir = data_dir.parent
    train_path = splits_dir / "train.csv"
    valid_path = splits_dir / "valid.csv"
    test_path = splits_dir / "test.csv"

    valid_splits = train_path.exists() and valid_path.exists() and test_path.exists()
    if Path(splits_dir).exists() and valid_splits and not force:
        logging.debug(f"Splits found in: {splits_dir} and none is missing.")
        return pd.read_csv(train_path), pd.read_csv(valid_path), pd.read_csv(test_path)

    logging.debug(
        f"Splits NOT found in: {splits_dir} or some are missing. Splitting..."
    )

    logging.debug(f"Identify unique {col_name}")
    unique = df[col_name].unique()

    logging.debug(
        f"Perform a {col_name}-level split Train / Valid / Test: 0.6 / 0.2 / 0.2"
    )
    train, valid_test = train_test_split(unique, test_size=0.4, random_state=42)
    valid, test = train_test_split(valid_test, test_size=0.5, random_state=42)

    logging.debug("Split the dataset")
    train_df = df[df[col_name].isin(train)]
    valid_df = df[df[col_name].isin(valid)]
    test_df = df[df[col_name].isin(test)]

    train_path.parent.mkdir(exist_ok=True, parents=True)
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    return pd.read_csv(train_path), pd.read_csv(valid_path), pd.read_csv(test_path)
