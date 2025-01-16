import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from .cross_models.cross_former import Crossformer
from torch.utils.data import DataLoader, Dataset
from data_processing.utils import EarlyStopping, adjust_learning_rate
from machine_learning_models.preprocessing import (
    load_data,
    create_lagged_features,
    preprocess_data,
    create_sequences,
    train_test_split_time_series,
    fill_na_values,
    extract_date_features,
)
from torch.amp import autocast, GradScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length, :-1]
        y = self.data[idx + self.sequence_length, -1]  # Single-step target
        return x, y

class CrossformerStockModel:
    def __init__(self, file_path, stock_name, hyperparameters=None):
        self.file_path = file_path
        self.stock_name = stock_name
        self.hyperparameters = hyperparameters or {
            "d_model": 64,
            "d_ff": 256,
            "n_heads": 4,
            "e_layers": 1,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32,
            "train_epochs": 50,
            "patience": 5,
            "seg_len": 6,
            "win_size": 3,
            "factor": 10,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 30

        # Load and preprocess data
        self.data = load_data(self.file_path)

        self.target = self.data["Close"]
        self.features = create_lagged_features(self.data)
        self.features = fill_na_values(self.features)
        self.features = extract_date_features(self.features)
        self.features = self.features.drop(columns=["Close"], errors="ignore")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(
            self.features, self.target
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split_time_series(
            self.X_train, self.y_train
        )

        (
            self.X_train,
            self.X_test,
            self.X_val,
            self.y_train,
            self.y_test,
            self.y_val,
            self.feature_scaler,
            self.target_scaler,
        ) = preprocess_data(
            self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val, add_feature_dim=False
        )

        # Add the last 29 rows (sequence length) from the train data to create sequences
        self.X_test = np.vstack([self.X_train[-self.sequence_length:], self.X_test])
        self.y_test = np.concatenate([self.y_train[-self.sequence_length:], self.y_test])

        # Flatten target arrays to ensure they are 1D
        self.y_train = self.y_train.ravel()
        self.y_test = self.y_test.ravel()
        self.y_val = self.y_val.ravel()

        # Combine features and target for sequence creation
        train_data = np.hstack([self.X_train, self.y_train.reshape(-1, 1)])
        test_data = np.hstack([self.X_test, self.y_test.reshape(-1, 1)])
        val_data = np.hstack([self.X_val, self.y_val.reshape(-1, 1)])

        self.train_dataset = TimeSeriesDataset(train_data, self.sequence_length)
        self.val_dataset = TimeSeriesDataset(val_data, self.sequence_length)
        self.test_dataset = TimeSeriesDataset(test_data, self.sequence_length)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.hyperparameters["batch_size"], shuffle=True, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.hyperparameters["batch_size"], shuffle=False, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.hyperparameters["batch_size"], shuffle=False, pin_memory=True
        )
        self._build_model()

    def _build_model(self):
        input_dim = self.X_train.shape[1]

        # Ensure out_len=1 for single-step prediction
        self.model = Crossformer(
            data_dim=input_dim,
            in_len=self.sequence_length,
            out_len=1,  # Single-step prediction
            seg_len=self.hyperparameters["seg_len"],
            win_size=self.hyperparameters["win_size"],
            factor=self.hyperparameters["factor"],
            d_model=self.hyperparameters["d_model"],
            d_ff=self.hyperparameters["d_ff"],
            n_heads=self.hyperparameters["n_heads"],
            e_layers=self.hyperparameters["e_layers"],
            dropout=self.hyperparameters["dropout"],
            baseline=False,
            device=self.device,
        ).float()
        self.model = self.model.to(self.device)

        if torch.cuda.device_count() > 1:
            print(f"[Model Init] Using {torch.cuda.device_count()} GPUs.")
            self.model = nn.DataParallel(self.model)

    def train(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.hyperparameters["learning_rate"])
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=self.hyperparameters["patience"], verbose=True)

        checkpoint_dir = os.path.join("checkpoints", self.stock_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        scaler = GradScaler()

        print("[Training] Starting training loop...")
        for epoch in range(self.hyperparameters["train_epochs"]):
            self.model.train()
            train_losses = []
            
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                model_optim.zero_grad()

                # Mixed precision
                with autocast(device_type="cuda"):  # Automatically uses float16 precision for computations
                    pred = self.model(batch_x)
                    pred = pred[:, :, 0].squeeze(-1)

                    loss = criterion(pred, batch_y)
                
                # Scales the loss and calls backward
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()

                train_losses.append(loss.item())

            train_loss_mean = sum(train_losses) / len(train_losses)
            val_loss = self._validate(criterion)

            print(
                f"[Epoch {epoch + 1}/{self.hyperparameters['train_epochs']}] "
                f"Train Loss: {train_loss_mean:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            early_stopping(val_loss, self.model, os.path.join(checkpoint_dir, "checkpoint.pth"))
            if early_stopping.early_stop:
                print("[Training] Early stopping triggered.")
                break

            # Adjust learning rate
            adjust_learning_rate(model_optim, epoch + 1, self.hyperparameters["learning_rate"], lradj='type1')

    def _validate(self, criterion):
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred = self.model(batch_x)

                pred = pred[:, :, 0].squeeze(-1)

                loss = criterion(pred, batch_y)
                val_losses.append(loss.item())

        return sum(val_losses) / len(val_losses)

    def _process_one_batch(self, batch_x, batch_y):
        """
        Processes a single batch of data.
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        outputs = self.model(batch_x)
        return outputs, batch_y

    def predict(self):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_x, _ in self.test_loader:
                batch_x = batch_x.float().to(self.device)

                pred = self.model(batch_x)

                pred = pred[:, :, 0].squeeze(-1)

                predictions.append(pred.cpu().numpy())

        predictions = self.target_scaler.inverse_transform(np.concatenate(predictions, axis=0).reshape(-1,1)).flatten()
        return predictions

    def save_model(self):
        """
        Saves the entire trained PyTorch Crossformer model to disk.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_crossformer_model.pth")
        
        # Save the entire model
        torch.save(self.model, model_path)
        print(f"Entire model saved to {model_path}")

    def save_predictions(self, predictions):
        """
        Saves the predictions as a CSV file.
        """
        prediction_dir = "Output_Data/saved_predictions"
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, f"Crossformer_{self.stock_name}_predictions.csv")

        prediction_dates = self.data.index[-len(predictions):]
        prediction_df = pd.DataFrame({"Date": prediction_dates, "Predicted Close": predictions.flatten()})
        prediction_df.to_csv(prediction_path, index=False)
        print(f"Predictions saved to {prediction_path}")

    def run(self):
        """
        Runs the training, testing, and saves the results.
        """
        print(f"Training Crossformer model for {self.stock_name}...")
        self.train()
        predictions = self.predict()
        self.save_model()
        self.save_predictions(predictions)
        return predictions
