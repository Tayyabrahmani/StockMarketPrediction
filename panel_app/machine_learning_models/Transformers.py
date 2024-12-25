import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
import pandas as pd
import numpy as np
from machine_learning_models.preprocessing import (
    load_data,
    create_lagged_features,
    preprocess_data,
    train_test_split_time_series,
)
from machine_learning_models.evaluation import evaluate_predictions, predict_and_inverse_transform


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, sequence_length, embed_dim=32, num_heads=2, ff_dim=128, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, embed_dim))
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        for layer in self.transformer_layers:
            x = layer(x)
        return self.fc(x[:, -1, :])  # Output from the last time step


class TransformerStockModel:
    def __init__(self, file_path, stock_name, hyperparameters=None):
        self.file_path = file_path
        self.stock_name = stock_name
        self.hyperparameters = hyperparameters or {
            "embed_dim": 32,
            "num_heads": 2,
            "ff_dim": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10
        }
        self.model = None
        self.scaler = None

        # Load and preprocess data
        self.data = load_data(self.file_path)
        self.data = create_lagged_features(self.data, target_col="Close")
        self.features, self.target, self.scaler = preprocess_data(self.data, target_col="Close")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(
            self.features, self.target
        )

    def build_model(self, input_dim, sequence_length):
        """
        Builds and initializes the Transformer model.
        """
        self.model = TransformerModel(
            input_dim=input_dim,
            sequence_length=sequence_length,
            embed_dim=self.hyperparameters["embed_dim"],
            num_heads=self.hyperparameters["num_heads"],
            ff_dim=self.hyperparameters["ff_dim"],
            num_layers=self.hyperparameters["num_layers"],
            dropout=self.hyperparameters["dropout"]
        )
        return self.model

    def train(self):
        """
        Trains the Transformer model.
        """
        train_loader = self._get_data_loader(self.X_train, self.y_train)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters["learning_rate"])
        criterion = nn.MSELoss()

        for epoch in range(self.hyperparameters["epochs"]):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.hyperparameters['epochs']}, Loss: {epoch_loss / len(train_loader):.4f}")

    def _get_data_loader(self, X, y):
        """
        Creates a DataLoader for the given data.
        """
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        )
        return DataLoader(dataset, batch_size=32, shuffle=False)

    def predict(self):
        """
        Generates predictions for the test data and maps them back to the original stock price range.

        Returns:
            np.array: Predicted values for the test data in the original stock price range.
        """
        predictions = predict_and_inverse_transform(self.model, self.X_test, self.scaler, feature_dim=self.X_test.shape[2])
        return predictions

    def save_model(self):
        """
        Saves the trained Transformer model to disk.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_transformer_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def save_predictions(self, predictions):
        """
        Saves the predictions as a CSV file.

        Parameters:
            predictions (np.array): Predicted values for the test data.
        """
        prediction_dir = "Output_Data/saved_predictions"
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, f"Transformer_{self.stock_name}_predictions.csv")

        # Save actual vs predicted values
        prediction_df = pd.DataFrame({
            "Date": pd.to_datetime(self.data.index[-len(predictions):]),
            "Predicted Close": predictions.flatten(),
        })
        prediction_df.to_csv(prediction_path, index=False)

    def run(self):
        """
        Runs the full pipeline: builds, trains, generates predictions, and saves the model and predictions.
        """
        print(f"Training Transformer model for {self.stock_name}...")
        sequence_length = self.features.shape[1]
        input_dim = self.X_train.shape[2]
        self.build_model(input_dim=input_dim, sequence_length=sequence_length)
        self.train()
        predictions = self.predict()
        self.save_model()
        self.save_predictions(predictions)
        return predictions
