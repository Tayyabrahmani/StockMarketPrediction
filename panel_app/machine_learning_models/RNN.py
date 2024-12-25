import torch
import torch.nn as nn
import os
import pickle
import pandas as pd
from machine_learning_models.preprocessing import (
    load_data,
    create_lagged_features,
    preprocess_data,
    train_test_split_time_series,
)
from machine_learning_models.evaluation import predict_and_inverse_transform


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class RNNStockModel:
    def __init__(self, file_path, stock_name, hyperparameters=None):
        """
        Initializes the RNNStockModel.

        Parameters:
            file_path (str): Path to the data file.
            stock_name (str): Name of the stock.
            hyperparameters (dict): Hyperparameters for training the RNN model.
        """
        self.file_path = file_path
        self.stock_name = stock_name
        self.hyperparameters = hyperparameters or {
            "hidden_dim": 32,
            "learning_rate": 0.001,
            "epochs": 10,
        }
        self.model = None
        self.scaler = None

        # Load and preprocess data
        self.data = load_data(self.file_path)
        self.data = create_lagged_features(self.data, target_col="Close")  # Add lagged features
        self.features, self.target, self.scaler = preprocess_data(self.data, target_col="Close")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(
            self.features, self.target
        )

    def build_model(self, input_dim):
        """
        Builds and initializes the RNN model.

        Parameters:
            input_dim (int): Number of input features.
        """
        self.model = RNN(input_dim, hidden_dim=self.hyperparameters["hidden_dim"])
        return self.model

    def train(self):
        """
        Trains the RNN model.
        """
        train_loader = self._get_data_loader(self.X_train, self.y_train)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters["learning_rate"])
        criterion = nn.MSELoss()

        for epoch in range(self.hyperparameters["epochs"]):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                # Reshape y_batch to match predictions' shape
                y_batch = y_batch.view(-1, 1)  # Reshape to [batch_size, 1]

                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.hyperparameters['epochs']}, Loss: {epoch_loss / len(train_loader):.4f}")

    def _get_data_loader(self, X, y):
        """
        Creates a DataLoader for the given data.

        Parameters:
            X (np.array): Input features.
            y (np.array): Target values.

        Returns:
            DataLoader: A PyTorch DataLoader.
        """
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )
        return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    def predict(self):
        """
        Generates predictions for the test data and inverse transforms them.

        Returns:
            np.array: Predictions in the original scale.
        """
        predictions = predict_and_inverse_transform(
            model=self.model,
            X=self.X_test,
            scaler=self.scaler,
            feature_dim=self.X_test.shape[2],  # Dynamically fetch feature dimension
        )
        return predictions

    def save_model(self):
        """
        Saves the trained model to disk.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_rnn_model.pkl")
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
        prediction_path = os.path.join(prediction_dir, f"RNN_{self.stock_name}_predictions.csv")

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
        print(f"Training RNN model for {self.stock_name}...")
        input_dim = self.X_train.shape[2]
        self.build_model(input_dim=input_dim)
        self.train()
        predictions = self.predict()
        self.save_model()
        self.save_predictions(predictions)
        return predictions
