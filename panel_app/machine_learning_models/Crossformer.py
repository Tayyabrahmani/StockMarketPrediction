import os
import json
import pickle
import optuna
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
from torch.cuda.amp import autocast, GradScaler

class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.9):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        errors = y_true - y_pred
        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)
        return loss.mean()

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

# class DirectionalLoss(nn.Module):
#     def forward(self, y_pred, y_true):
#         direction_true = torch.sign(y_true[1:] - y_true[:-1])
#         direction_pred = torch.sign(y_pred[1:] - y_pred[:-1])
#         directional_error = (direction_true != direction_pred).float()
#         return directional_error.mean()

class CrossformerStockModel:
    def __init__(self, file_path, stock_name, hyperparameters=None):
        self.file_path = file_path
        self.stock_name = stock_name
        self.hyperparameters = hyperparameters or {
            "d_model": 256,
            "d_ff": 512,
            "n_heads": 8,
            "e_layers": 2,
            "dropout": 0.1,
            # "learning_rate": 0.003902310852880716,
            # "learning_rate": 0.00019216000054779,
            "learning_rate": 0.003,
            "batch_size": 32,
            "train_epochs": 100,
            "patience": 5,
            "seg_len": 8,
            "win_size": 2,
            "factor": 10,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 20

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

        # Add the last 29 rows (sequence length) from the train data to create sequences
        self.X_test = np.vstack([self.X_train[-self.sequence_length:], self.X_test])
        self.y_test = np.concatenate([self.y_train[-self.sequence_length:], self.y_test])

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split_time_series(
            self.X_train, self.y_train, test_size=0.1
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
        model_optim = optim.AdamW(self.model.parameters(), lr=self.hyperparameters["learning_rate"], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(model_optim, max_lr=0.0005, steps_per_epoch=len(self.train_loader), epochs=self.hyperparameters["train_epochs"])
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, mode='min', factor=0.5, patience=3, verbose=True)

        criterion_quantile = nn.MSELoss()
        # criterion_quantile = QuantileLoss(quantile=0.9)
        # criterion_directional = DirectionalLoss()
        early_stopping = EarlyStopping(patience=self.hyperparameters["patience"], verbose=True)

        checkpoint_dir = os.path.join("checkpoints", self.stock_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        scaler = GradScaler()

        print("[Training] Starting training loop...")
        for epoch in range(self.hyperparameters["train_epochs"]):
            self.model.train()
            train_losses = []
            
            for batch_x, batch_y in self.train_loader:
                model_optim.zero_grad()
                pred, true = self._process_one_batch(batch_x, batch_y)
                loss = criterion_quantile(pred, true)

                # Scales the loss and calls backward
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()

                train_losses.append(loss.item())

            train_loss_mean = sum(train_losses) / len(train_losses)
            # val_loss = self._validate_with_multiple_criteria(self.val_loader, criterion_quantile, criterion_directional)
            val_loss = self._validate_with_multiple_criteria(self.val_loader, criterion_quantile)

            print(
                f"[Epoch {epoch + 1}/{self.hyperparameters['train_epochs']}] "
                f"Train Loss: {train_loss_mean:.4f} | Val Loss: {val_loss:.4f}"
            )

            scheduler.step()

            # Early stopping
            early_stopping(val_loss, self.model, os.path.join(checkpoint_dir, "checkpoint.pth"))
            if early_stopping.early_stop:
                print("[Training] Early stopping triggered.")
                break

            # # Adjust learning rate
            # adjust_learning_rate(model_optim, epoch + 1, self.hyperparameters["learning_rate"], lradj='type1')

    # def _validate_with_multiple_criteria(self, val_loader, criterion_quantile, criterion_directional):
    def _validate_with_multiple_criteria(self, val_loader, criterion_quantile):
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred, true = self._process_one_batch(batch_x, batch_y)  # Use function

                # Compute individual losses
                loss_quantile = criterion_quantile(pred, true)

                # loss_directional = criterion_directional(pred, batch_y)
                # total_loss = loss_quantile + 0.1 * loss_directional
                total_loss = loss_quantile

                val_losses.append(total_loss.item())

        return sum(val_losses) / len(val_losses)

    def _process_one_batch(self, batch_x, batch_y, inverse=False):
        """
        Processes a single batch: moves tensors to GPU, passes input through model, extracts last timestep,
        and applies inverse transformation (if needed).
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        # TODO: Remove after testing
        # batch_x += torch.randn_like(batch_x) * 0.01
        
        outputs = self.model(batch_x)  # Get model predictions (batch_size, seq_len, output_dim)
        # Extract last time step prediction
        outputs = outputs[:, -1, 0]  # Take the last timestep from the first output dimension
        # batch_y = batch_y[:, -1]  # Take the corresponding target value

        # Apply inverse transformation if needed
        if inverse:
            outputs = self.target_scaler.inverse_transform(outputs.cpu().detach().numpy()).flatten()
            batch_y = self.target_scaler.inverse_transform(batch_y.cpu().detach().numpy()).flatten()

        return outputs, batch_y

    def optimize_hyperparameters(self, n_trials=50):
        """
        Optimize hyperparameters using Optuna and update self.hyperparameters.

        Parameters:
            n_trials (int): Number of trials to run for optimization.

        Returns:
            dict: Best hyperparameters found during optimization.
        """
        def objective(trial):
            # Define the search space
            d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])
            d_ff = trial.suggest_int("d_ff", 128, 512, step=64)
            n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
            e_layers = trial.suggest_int("e_layers", 1, 4)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            seg_len = trial.suggest_int("seg_len", 4, 12, step=2)

            # Temporarily update self.hyperparameters for the trial
            self.hyperparameters = {
                "d_model": d_model,
                "d_ff": d_ff,
                "n_heads": n_heads,
                "e_layers": e_layers,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "train_epochs": 30,
                "patience": 5,
                "seg_len": seg_len,
                "win_size": 2,
                "factor": 10,
            }

            # Build and validate the model
            self._build_model()
            train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
            )
            val_loader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
            )
            criterion_mse = nn.MSELoss()
            # criterion_directional = DirectionalLoss()
            model_optim = optim.AdamW(self.model.parameters(), lr=learning_rate)
            scaler = GradScaler()

            for epoch in range(self.hyperparameters["train_epochs"]):
                self.model.train()
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    model_optim.zero_grad()
                    with autocast():
                        pred, true = self._process_one_batch(batch_x, batch_y)
                        loss_mse = criterion_mse(pred, true)
                        # loss_directional = criterion_directional(pred, batch_y)

                        # loss = loss_mse + 0.1 * loss_directional
                        loss = loss_mse

                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()

                # val_loss = self._validate_with_multiple_criteria(self.val_loader, criterion_mse, criterion_directional)
                val_loss = self._validate_with_multiple_criteria(self.val_loader, criterion_mse)
                if epoch >= self.hyperparameters["patience"]:
                    break

            return val_loss

        # Run Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        # Update self.hyperparameters with the best ones
        self.hyperparameters.update(study.best_params)
        print("Best Hyperparameters Found:", study.best_params)

    def predict(self):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                pred, true = self._process_one_batch(batch_x, batch_y)            
                predictions.append(pred.cpu().numpy())

        raw_predictions = np.concatenate(predictions, axis=0)
        predictions = self.target_scaler.inverse_transform(raw_predictions.reshape(-1,1)).flatten()

        # # 🔥 Apply correction factor
        # correction_factor = np.mean(self.y_train) / np.mean(predictions)
        # predictions = predictions * correction_factor  

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
        prediction_path = os.path.join(prediction_dir, f"Crossformers_{self.stock_name}_predictions.csv")

        prediction_dates = self.data.index[-len(predictions):]
        prediction_df = pd.DataFrame({"Date": prediction_dates, "Predicted Close": predictions.flatten()})
        prediction_df.to_csv(prediction_path, index=False)
        print(f"Predictions saved to {prediction_path}")

    def save_hyperparameters(self):
        """
        Saves the hyperparameters as a CSV file.
        """
        hyperparam_dir = os.path.join("Output_Data", "Hyperparameters", "Crossformer")
        os.makedirs(hyperparam_dir, exist_ok=True)
        hyperparam_path = os.path.join(hyperparam_dir, f"{self.stock_name}_hyperparameter.csv")

        hyperparam_df = pd.DataFrame.from_dict(self.hyperparameters, orient="index", columns=["Value"])
        hyperparam_df.to_csv(hyperparam_path)
        print(f"Hyperparameters saved to {hyperparam_path}")

    def run(self):
        """
        Runs the training, testing, and saves the results.
        """
        # print(f"Optimizing Hyperparameters")
        # self.optimize_hyperparameters(n_trials=30)
        print(f"Training Crossformer model for {self.stock_name}...")
        self.train()
        predictions = self.predict()
        self.save_model()
        self.save_predictions(predictions)
        # self.save_hyperparameters()
        return predictions
