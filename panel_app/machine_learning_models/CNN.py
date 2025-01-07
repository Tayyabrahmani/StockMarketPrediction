import torch
import torch.nn as nn
import os
import pickle
import pandas as pd
from machine_learning_models.preprocessing import (
    load_data,
    create_lagged_features,
    preprocess_data,
    create_sequences,
    train_test_split_time_series,
    fill_na_values,
    extract_date_features
)
from machine_learning_models.evaluation import predict_and_inverse_transform
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna

class CNN(nn.Module):
    def __init__(self, input_dim, sequence_length, num_filters=64, dropout_rate=0.2):
        super(CNN, self).__init__()

        # Convolutional layers with residual connections
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.conv3 = nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_filters * 4)
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.flattened_size = num_filters * 4 * sequence_length
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        batch_size = x.size(0)

        # CNN Feature Extraction
        x = x.permute(0, 2, 1).contiguous()
        residual = x.reshape(batch_size, -1)  # Flatten input for residual connection

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        # Flatten for fully connected layers
        x = x.reshape(batch_size, -1)

        # Align residual dimensions
        if residual.size(1) != x.size(1):
            residual = nn.Linear(residual.size(1), x.size(1)).to(x.device)(residual)

        # Residual connection
        x += residual

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class CNNStockModel:
    def __init__(self, file_path, stock_name):
        self.file_path = file_path
        self.stock_name = stock_name
        self.model = None

        # Load and preprocess data
        self.data = load_data(self.file_path)

        # Create lagged features
        self.data = create_lagged_features(self.data, target_col="Close")
        self.data = fill_na_values(self.data)
        self.data = extract_date_features(self.data)

        X, y = create_sequences(self.data, sequence_length=30, target_col="Close")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(
            X, y
        )
        self.X_train, self.X_test, self.scaler = preprocess_data(self.X_train, self.X_test)

    def objective(self, trial):
        # Suggest hyperparameters for Optuna to tune
        num_filters = trial.suggest_categorical("num_filters", [16, 32, 64])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 10, 30)

        # Initialize and train the model
        sequence_length = self.X_train.shape[1]
        input_dim = self.X_train.shape[2]
        model = CNN(input_dim=input_dim, sequence_length=sequence_length, dropout_rate=dropout_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(-1)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # Validation loss calculation
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32).unsqueeze(-1)
            predictions = model(X_test_tensor)
            val_loss = criterion(predictions, y_test_tensor).item()

        return val_loss

    def run_tuning(self, n_trials=15, update_optuna_study=False):
        if not update_optuna_study:
            return

        # Define a pruner for early stopping of unpromising trials
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

        # Create the Optuna study with the pruner
        study = optuna.create_study(direction="minimize", pruner=pruner)

        # Optimize the study
        study.optimize(self.objective, n_trials=n_trials)

        print("Best hyperparameters:", study.best_params)
        print("Best validation loss:", study.best_value)
        return study.best_params

    def train(self, learning_rate=1e-3, epochs=10, batch_size=32):
        """
        Trains the CNN model.
        """
        # Prepare data for PyTorch DataLoader
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(-1)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define loss and optimizer
        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)

    def predict(self):
        """
        Predicts using the trained EnhancedCNN model.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Build and train the model before predicting.")

        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
            predictions = self.model(X_test_tensor).numpy()

        predictions_reshaped = predictions.reshape(-1, 1)
        return predictions

        # return predict_and_inverse_transform(
        #     model=self.model,
        #     X=predictions_reshaped,
        #     scaler=self.scaler,
        #     feature_dim=self.X_test.shape[2]
        # )

    def save_model(self):
        """
        Saves the trained CNN model as a pickle file.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_cnn_model.pkl")
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
        prediction_path = os.path.join(prediction_dir, f"CNN_{self.stock_name}_predictions.csv")

        # Save actual vs predicted values
        prediction_df = pd.DataFrame({
            "Date": pd.to_datetime(self.data.index[-len(predictions):]),
            "Predicted Close": predictions.flatten()
        })
        prediction_df.to_csv(prediction_path, index=False)

    def run(self):
        """
        Runs the full pipeline: trains the model, generates predictions, and saves the model and predictions.
        """
        # best_params = self.run_tuning(update_optuna_study=False)
        best_params = {'num_filters': 32, 'dropout_rate': 0.20815363258412045, 'learning_rate': 0.0005816740646783397, 'epochs': 1}
        print(f"Best parameters found: {best_params}")

        sequence_length = self.X_train.shape[1]
        input_dim = self.X_train.shape[2]

        self.model = CNN(input_dim=input_dim,
                         sequence_length=sequence_length,
                         num_filters=best_params["num_filters"],
                         dropout_rate=best_params["dropout_rate"],
        )

        print("Training model...")
        self.train(learning_rate=best_params["learning_rate"], epochs=best_params["epochs"])

        print("Generating predictions...")
        predictions = self.predict()

        print("Saving predictions...")
        self.save_predictions(predictions)

        print("Saving the model...")
        self.save_model()
        
        return predictions