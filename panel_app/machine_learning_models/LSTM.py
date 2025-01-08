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
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def build_model(self, input_dim, hidden_dim):
        """
        Builds and initializes the LSTM model.

        Parameters:
            input_dim (int): Number of input features.
        """
        self.model = LSTM(input_dim, hidden_dim=hidden_dim)
        return self.model

    def train(self, batch_size, learning_rate, epochs):
        """
        Trains the LSTM model.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() before training.")

        # Prepare data for PyTorch DataLoader
        train_loader = self._get_data_loader(self.X_train, self.y_train, batch_size)

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Move model to device (GPU if available)
        self.model.to(self.device)

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
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

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
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def predict(self):
        """
        Generates predictions for the test data and inverse transforms them to the original scale.

        Returns:
            np.array: Predicted values for the test data in the original scale.
        """
        self.model.eval()

        # Ensure the test data is on the correct device
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy()

        return predictions

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
            "Predicted Close": predictions.flatten(),
        })
        prediction_df.to_csv(prediction_path, index=False)

    def run(self, batch_size=16, learning_rate=0.001, hidden_dim=150, epochs=50):
        """
        Runs the full pipeline: trains the model, generates predictions, and saves the model and predictions.
        """
        input_dim = self.X_train.shape[2]
        print("Building the model...")
        self.build_model(input_dim=input_dim, hidden_dim=hidden_dim)

        print("Training the model...")
        self.train(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs)

        print("Generating predictions...")
        predictions = self.predict()

        print("Saving the model and predictions...")
        self.save_model()
        self.save_predictions(predictions)

        return predictions
