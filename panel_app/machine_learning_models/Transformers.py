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
    fill_na_values,
    extract_date_features
)
from machine_learning_models.evaluation import predict_and_inverse_transform

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-np.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, embed_size, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.input_embedding = nn.Linear(1, embed_size)  # Embedding for 1D input (stock prices)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(embed_size, 1)

    def forward(self, src, tgt):
        """
        Forward pass for the Transformer.

        Parameters:
            src (torch.Tensor): Source sequence (shape: batch_size, src_len, feature_dim).
            tgt (torch.Tensor): Target sequence (shape: batch_size, tgt_len, feature_dim).

        Returns:
            torch.Tensor: Output predictions for the target sequence.
        """
        # Add feature dimension to src and tgt
        src = src.unsqueeze(-1)  # Shape: (batch_size, src_len, 1)
        tgt = tgt.unsqueeze(-1)  # Shape: (batch_size, tgt_len, 1)

        # Embed and encode positional information
        src = self.input_embedding(src)  # Shape: (batch_size, src_len, embed_size)
        src = self.positional_encoding(src)

        tgt = self.input_embedding(tgt)  # Shape: (batch_size, tgt_len, embed_size)
        tgt = self.positional_encoding(tgt)

        # Pass through the transformer
        output = self.transformer(src, tgt)

        # Return the prediction for the last time step
        return self.fc_out(output[:, -1, :])

class TransformerStockModel:
    def __init__(self, file_path, stock_name, hyperparameters=None):
        self.file_path = file_path
        self.stock_name = stock_name
        self.hyperparameters = hyperparameters or {
            "embed_size": 32,
            "num_heads": 4,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "dropout": 0.1,
            "learning_rate": 0.0001,
            "epochs": 30
        }
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load and preprocess data
        self.data = load_data(self.file_path)

        # Create features and target dataframes
        self.target = self.data["Close"]
        self.features = create_lagged_features(self.data)
        self.features = fill_na_values(self.features)
        self.features = extract_date_features(self.features)
        self.features = self.features.drop(columns=['Close'], errors='ignore')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(
            self.features, self.target
        )
        self.X_train, self.X_test, self.y_train, self.y_test, self.feature_scaler, self.target_scaler = preprocess_data(self.X_train, self.X_test, self.y_train, self.y_test)

    def build_model(self):
        """
        Builds and initializes the Transformer model.
        """
        self.model = TransformerModel(
            embed_size=self.hyperparameters["embed_size"],
            num_heads=self.hyperparameters["num_heads"],
            num_encoder_layers=self.hyperparameters["num_encoder_layers"],
            num_decoder_layers=self.hyperparameters["num_decoder_layers"],
            dropout=self.hyperparameters["dropout"]
        )
        return self.model

    def train(self, batch_size):
        """
        Trains the Transformer model.
        """
        train_loader = self._get_data_loader(self.X_train, self.y_train, batch_size=batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters["learning_rate"])
        criterion = nn.MSELoss()

        # Move model to device (GPU if available)
        self.model.to(self.device)

        for epoch in range(self.hyperparameters["epochs"]):
            self.model.train()
            epoch_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                src = X_batch[:, :-1]
                tgt = X_batch[:, 1:]

                optimizer.zero_grad()
                predictions = self.model(src, tgt)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.hyperparameters['epochs']}, Loss: {epoch_loss / len(train_loader):.4f}")

    def _get_data_loader(self, X, y, batch_size):
        """
        Creates a DataLoader for the given data.
        """
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32).to(self.device), 
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def predict(self):
        """
        Generates predictions for the test data and maps them back to the original stock price range.

        Returns:
            np.array: Predicted values for the test data in the original stock price range.
        """
        self.model.eval()

        # Ensure the test data is on the correct device
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        src = X_test_tensor[:, :-1]  # Input sequence excluding the last value
        tgt = X_test_tensor[:, 1:]   # Placeholder target sequence

        with torch.no_grad():
            predictions = self.model(src, tgt).cpu().numpy()

        # Concatenate predictions and inverse transform
        predictions_original_scale = self.target_scaler.inverse_transform(predictions)
        return predictions_original_scale.flatten()

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
        self.build_model()
        self.train(batch_size=16)
        predictions = self.predict()
        self.save_model()
        self.save_predictions(predictions)
        return predictions
