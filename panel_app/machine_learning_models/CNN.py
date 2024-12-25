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


class CNN(nn.Module):
    def __init__(self, input_dim, sequence_length, num_filters=16, kernel_size=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear((sequence_length // 2) * num_filters, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNNStockModel:
    def __init__(self, file_path, stock_name, hyperparameters=None):
        self.file_path = file_path
        self.stock_name = stock_name
        self.hyperparameters = hyperparameters or {"num_filters": 16, "learning_rate": 0.001, "epochs": 10}
        self.model = None
        self.scaler = None

        # Load and preprocess data
        self.data = load_data(self.file_path)

        # Create lagged features
        self.data = create_lagged_features(self.data, target_col="Close")

        self.features, self.target, self.scaler = preprocess_data(self.data, target_col="Close")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(
            self.features, self.target
        )

    def build_model(self, input_dim, sequence_length):
        """
        Builds the CNN model.

        Parameters:
            input_dim (int): Number of input features.
            sequence_length (int): Length of the input sequence.

        Returns:
            CNN: The initialized CNN model.
        """
        self.model = CNN(
            input_dim=input_dim,
            sequence_length=sequence_length,
            num_filters=self.hyperparameters["num_filters"],
        )
        return self.model

    def train(self):
        """
        Trains the CNN model.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() before training.")

        # Prepare data for PyTorch DataLoader
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(-1)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters["learning_rate"])

        # Training loop
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

    def predict(self):
        """
        Generates predictions for the test data.

        Returns:
            np.array: Predicted values for the test data.
        """
        sequence_length = self.features.shape[1]
        predictions = predict_and_inverse_transform(
            self.model,
            self.X_test,
            scaler=self.scaler,
            feature_dim=self.X_test.shape[2],
        )
        return predictions

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
        sequence_length = self.features.shape[1]
        input_dim = self.X_train.shape[2]
        self.build_model(input_dim=input_dim, sequence_length=sequence_length)
        self.train()
        predictions = self.predict()
        self.save_model()
        self.save_predictions(predictions)
        return predictions                                                                                                     