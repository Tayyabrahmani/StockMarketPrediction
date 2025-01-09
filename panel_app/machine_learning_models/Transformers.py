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
    preprocess_data_svr,
    train_test_split_time_series,
    fill_na_values,
    extract_date_features
)
from machine_learning_models.evaluation import predict_and_inverse_transform

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        dec_seq_len,
        max_seq_len,
        out_seq_len,
        d_model,
        n_encoder_layers,
        n_decoder_layers,
        n_heads,
        dim_feedforward,
        dropout=0.1,
    ):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_input_layer = nn.Linear(input_size, d_model)
        self.decoder_input_layer = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoder(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Final output layer
        self.linear_mapping = nn.Linear(d_model * out_seq_len, out_seq_len)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src.ndim == 2:
            src = src.unsqueeze(-1)  # Add a feature dimension if missing

        src = self.encoder_input_layer(src)
        src = self.positional_encoding(src)

        encoder_output = self.encoder(src, src_key_padding_mask=src_mask)

        tgt = self.decoder_input_layer(tgt)
        tgt = self.positional_encoding(tgt)

        decoder_output = self.decoder(
            tgt=tgt, memory=encoder_output, tgt_mask=tgt_mask, memory_mask=src_mask
        )

        # decoder_output = self.linear_mapping(decoder_output.flatten(start_dim=1))
        decoder_output = self.linear_mapping(decoder_output[:, -1, :])
        return decoder_output

class TransformerStockModel:
    def __init__(self, file_path, stock_name, hyperparameters=None):
        self.file_path = file_path
        self.stock_name = stock_name
        self.hyperparameters = hyperparameters or {
            "dec_seq_len": 30,
            "max_seq_len": 500,
            "d_model": 64,
            "n_encoder_layers": 4,
            "n_decoder_layers": 4,
            "n_heads": 8,
            "dim_feedforward": 256,
            "dropout": 0.2,
            "learning_rate": 0.0001,
            "epochs": 30, 
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
        # self.X_train, self.X_test, self.y_train, self.y_test, self.feature_scaler, self.target_scaler = preprocess_data(self.X_train, self.X_test, self.y_train, self.y_test, add_feature_dim=True)
        self.X_train, self.X_test, self.y_train, self.y_test, self.feature_scaler, self.target_scaler = preprocess_data_svr(self.X_train, self.X_test, self.y_train, self.y_test)
        
    def build_model(self):
        """
        Builds and initializes the Transformer model.
        """
        input_size = self.X_train.shape[-1]
        out_seq_len = self.X_test.shape[-1]

        self.model = TimeSeriesTransformer(
            input_size=input_size,
            dec_seq_len=self.hyperparameters["dec_seq_len"],
            max_seq_len=self.hyperparameters["max_seq_len"], 
            out_seq_len=out_seq_len,
            d_model=self.hyperparameters["d_model"],
            n_encoder_layers=self.hyperparameters["n_encoder_layers"],
            n_decoder_layers=self.hyperparameters["n_decoder_layers"], 
            n_heads=self.hyperparameters["n_heads"], 
            dim_feedforward=self.hyperparameters["dim_feedforward"], 
            dropout=self.hyperparameters["dropout"], 
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
                src = X_batch[:, :-1,]  # Shape: (batch_size, seq_len, num_features)
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
