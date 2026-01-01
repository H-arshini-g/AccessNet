# src/train_model.py
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, model_type="lstm"):
        super(RNNModel, self).__init__()
        self.model_type = model_type.lower()
        if self.model_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif self.model_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Invalid model type. Use 'lstm' or 'gru'.")
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out


def train_model(features_file, model_type="lstm", hidden_size=32, num_layers=1, epochs=20,
                lr=0.001, batch_size=4, out_dir="data/outputs"):
    # Load features
    data = np.load(features_file)
    X, y = data["X"], data["y"]

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = X.shape[2]   # number of stops
    output_size = y.shape[1]  # number of stops
    model = RNNModel(input_size, hidden_size, num_layers, output_size, model_type)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    # Save model
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{model_type}_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"{model_type.upper()} model saved to {model_path}")

    # Save predictions for visualization
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).numpy()

    # Save to CSV
    times = np.arange(len(preds))  # dummy timeline
    df_preds = pd.DataFrame(preds, columns=[f"stop_{i}" for i in range(output_size)])
    df_preds.insert(0, "time_step", times)
    preds_path = os.path.join(out_dir, f"{model_type}_preds.csv")
    df_preds.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", default="data/outputs")
    args = parser.parse_args()

    train_model(args.features, model_type=args.model, epochs=args.epochs, out_dir=args.out)
