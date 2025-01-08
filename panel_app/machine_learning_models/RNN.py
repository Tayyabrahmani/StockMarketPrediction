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


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=3, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class RNNStockModel:
    def __init__(self, file_path, stock_name):
        """
        Initializes the RNNStockModel.

        Parameters:
            file_path (str): Path to the data file.
            stock_name (str): Name of the stock.
            hyperparameters (dict): Hyperparameters for training the RNN model.
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

    def build_model(self, input_dim, hidden_dim, num_layers, dropout):
        """
        Builds and initializes the RNN model.

        Parameters:
            input_dim (int): Number of input features.
        """
        self.model = RNN(input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(self.device)
        return self.model

    def train(self, batch_size, learning_rate, epochs):
        """
        Trains the RNN model.
        """
        train_loader = self._get_data_loader(self.X_train, self.y_train, batch_size)
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                # Move data to device
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()

                # # Reshape y_batch to match predictions' shape
                # y_batch = y_batch.view(-1, 1)  # Reshape to [batch_size, 1]

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
            DataLoader: A PyTorch DataLoader.
        """
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32).to(self.device), 
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

        return predictions

        # predictions = predict_and_inverse_transform(
        #     model=self.model,
        #     X=self.X_test,
        #     scaler=self.scaler,
        #     feature_dim=self.X_test.shape[2],  # Dynamically fetch feature dimension
        # )
        # return predictions

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

    def run(self, batch_size=16, learning_rate=0.0008, hidden_dim=150, num_layers=1, epochs=30, dropout=0.2):
        """
        Runs the full pipeline: builds, trains, generates predictions, and saves the model and predictions.
        """
        print(f"Training RNN model for {self.stock_name}...")
        input_dim = self.X_train.shape[2]
        self.build_model(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

        self.train(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs)
        predictions = self.predict()
        self.save_model()
        self.save_predictions(predictions)
        return predictions
