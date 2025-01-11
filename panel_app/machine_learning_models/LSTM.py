import torch
import torch.nn as nn
import os
import pickle
import pandas as pd
import numpy as np
from machine_learning_models.preprocessing import (
    load_data,
    create_lagged_features,
    preprocess_data,
    create_sequences,
    train_test_split_time_series,
    fill_na_values,
    extract_date_features
)
from machine_learning_models.evaluation import (
    predict_and_inverse_transform,
)
import optuna


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=150, num_layers=1, dropout=0.2):
        """
        Initializes the LSTM model.

        Parameters:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
            num_layers (int): Number of stacked LSTM layers.
        """
        super(LSTM, self).__init__()

        actual_dropout = 0.0 if num_layers == 1 else dropout

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=actual_dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=actual_dropout, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMStockModel:
    def __init__(self, file_path, stock_name):
        """
        Initializes the LSTMStockModel with the necessary preprocessing and hyperparameters.

        Parameters:
            file_path (str): Path to the input CSV file.
            stock_name (str): Name of the stock being analyzed.
            hyperparameters (dict, optional): Model hyperparameters.
        """
        self.file_path = file_path
        self.stock_name = stock_name
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 30

        # Load and preprocess data
        self.data = load_data(self.file_path)

        # Create lagged features
        self.data = create_lagged_features(self.data, target_col="Close")
        self.data = fill_na_values(self.data)
        self.data = extract_date_features(self.data)

        # X, y = create_sequences(self.data, sequence_length=30, target_col="Close")
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(
        #     X, y
        # )
        # self.X_train, self.X_test, self.y_train, self.y_test, self.feature_scaler, self.target_scaler = preprocess_data(self.X_train, self.X_test, self.y_train, self.y_test, add_feature_dim=True)

        # Create features and target dataframes
        self.target = self.data["Close"]
        self.features = create_lagged_features(self.data)
        self.features = fill_na_values(self.features)
        self.features = extract_date_features(self.features)
        self.features = self.features.drop(columns=['Close'], errors='ignore')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(
            self.features, self.target
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split_time_series(
            self.X_train, self.y_train
        )

        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val, self.feature_scaler, self.target_scaler = preprocess_data(self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val, add_feature_dim=False)

        # Add the last 29 rows (sequence length) from the train data to create sequences
        self.X_test = np.vstack([self.X_train[-self.sequence_length:], self.X_test])
        self.y_test = np.concatenate([self.y_train[-self.sequence_length:], self.y_test])

        # Concatenate features and targets for sequence creation (train)
        data_train = np.hstack([self.X_train, self.y_train.reshape(-1, 1)])
        self.X_train, self.y_train = create_sequences(
            data_train, sequence_length=self.sequence_length, target_col="Close", is_df=False
        )

        # Concatenate features and targets for sequence creation (test)
        data_test = np.hstack([self.X_test, self.y_test.reshape(-1, 1)])
        self.X_test, self.y_test = create_sequences(
            data_test, sequence_length=self.sequence_length, target_col="Close", is_df=False
        )

        data_val = np.hstack([self.X_val, self.y_val.reshape(-1, 1)])
        self.X_val, self.y_val = create_sequences(
            data_val, sequence_length=self.sequence_length, target_col="Close", is_df=False
        )

    def build_model(self, input_dim, hidden_dim, num_layers, dropout):
        """
        Builds and initializes the LSTM model.

        Parameters:
            input_dim (int): Number of input features.
        """
        self.model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        return self.model

    def train(self, batch_size, learning_rate, epochs, early_stop_patience):
        """
        Trains the LSTM model.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() before training.")

        # Prepare data for PyTorch DataLoader
        train_loader = self._get_data_loader(self.X_train, self.y_train, batch_size)
        val_loader = self._get_data_loader(self.X_val, self.y_val, batch_size)

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # Move model to device (GPU if available)
        self.model.to(self.device)
        best_val_loss = float('inf')
        epochs_no_improve = 0

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                # Move data to device
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    val_predictions = self.model(X_val)
                    val_loss += criterion(val_predictions, y_val).item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Save the best model
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print("Early stopping triggered.")
                    break

            scheduler.step()

    def _get_data_loader(self, X, y, batch_size):
        """
        Creates a DataLoader for the given data.

        Parameters:
            X (np.array): Input features.
            y (np.array): Target values.

        Returns:
            DataLoader: PyTorch DataLoader.
        """
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32).to(self.device), 
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def predict(self):
        """
        Generates predictions for the test data and inverse transforms them.

        Returns:
            np.array: Predictions in the original scale.
        """
        self.model.eval()

        # Ensure the test data is on the correct device
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy()

        predictions_original_scale = self.target_scaler.inverse_transform(predictions.reshape(-1, 1))
        
        return predictions_original_scale.flatten()

    def save_model(self):
        """
        Saves the trained LSTM model as a pickle file.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_lstm_model.pkl")
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
        prediction_path = os.path.join(prediction_dir, f"LSTM_{self.stock_name}_predictions.csv")

        # Save actual vs predicted values
        prediction_df = pd.DataFrame({
            "Date": pd.to_datetime(self.data.index[-len(predictions):]),
            "Predicted Close": predictions,
        })
        prediction_df.to_csv(prediction_path, index=False)

    def objective(self, trial):
        """
        Objective function for Optuna hyperparameter tuning.
        """
        # Define the hyperparameter search space
        hidden_dim = trial.suggest_int("hidden_dim", 50, 300)
        num_layers = trial.suggest_int("num_layers", 1, 5)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # Build and train the model
        self.build_model(input_dim=self.X_train.shape[2], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.train(batch_size=batch_size, learning_rate=learning_rate, epochs=50, early_stop_patience=10)

        # Evaluate on the validation set
        self.model.eval()
        X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
        y_val_actual = self.target_scaler.inverse_transform(self.y_val.reshape(-1, 1)).flatten()

        with torch.no_grad():
            val_predictions = self.model(X_val_tensor).cpu().numpy()

        val_predictions_original_scale = self.target_scaler.inverse_transform(val_predictions.reshape(-1, 1)).flatten()

        # Calculate RMSE on validation set
        rmse = np.sqrt(np.mean((val_predictions_original_scale - y_val_actual) ** 2))

        return rmse

    def tune_hyperparameters(self, n_trials=50):
        """
        Perform hyperparameter tuning using Optuna.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)

        print("Best trial:")
        print(study.best_trial)
        print("Best hyperparameters:")
        print(study.best_params)

        return study.best_params

    def run(self, batch_size=64, learning_rate=0.00036, num_layers=3, dropout=0.41, hidden_dim=137, epochs=150, early_stop_patience=20):
        """
        Runs the full pipeline: trains the model, generates predictions, and saves the model and predictions.
        """
        input_dim = self.X_train.shape[2]
        best_params=self.tune_hyperparameters(n_trials=50)

        print("Building the LSTM model...")
        # self.build_model(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.build_model(input_dim=input_dim, hidden_dim=best_params["hidden_dim"], num_layers=best_params["num_layers"], dropout=best_params["dropout"])

        print("Training the LSTM model...")
        # self.train(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, early_stop_patience=early_stop_patience)
        self.train(batch_size=best_params["batch_size"], learning_rate=best_params["learning_rate"], epochs=epochs, early_stop_patience=early_stop_patience)

        print("Generating predictions...")
        predictions = self.predict()

        print("Saving the model and predictions...")
        self.save_model()
        self.save_predictions(predictions)
        return predictions
