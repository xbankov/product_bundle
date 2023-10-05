import logging
import random
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from product_bundle.pricing import RegressionModel
from product_bundle.utils import load_product_df

random.seed(0)
torch.manual_seed(0)


def main():
    data_dir = Path("data")

    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    product_df = load_product_df(data_dir)
    X = torch.tensor(product_df.iloc[:, 8:-4].values, dtype=torch.float32)
    y = torch.tensor(product_df["UnitPrice"].values, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    model = RegressionModel(input_size=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 50
    for _ in tqdm(range(num_epochs)):
        model.train()
        for batch in train_dataloader:
            X_batch, y_batch = batch
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in test_dataloader:
            X_batch, y_batch = batch
            y_pred = model(X_batch)
            test_loss += criterion(y_pred.squeeze(), y_batch)

    torch.save(model.state_dict(), "static/pricing_model_weights.pth")
    print(f"Test MSE Loss: {test_loss / len(test_dataloader):.4f}")


if __name__ == "__main__":
    main()
