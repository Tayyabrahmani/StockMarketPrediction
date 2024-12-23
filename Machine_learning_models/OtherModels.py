import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

# Data Preprocessing
def preprocess_data(df, target_col, sequence_length=30):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df)
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-1])
        y.append(data[i + sequence_length, -1])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def create_lagged_features(df, n_lags=3):
    for lag in range(1, n_lags + 1):
        df[f"Close_Lag_{lag}"] = df['Close'].shift(lag)
        df[f"Volume_Lag_{lag}"] = df['Volume'].shift(lag)
    df.dropna(inplace=True)
    return df

# Split data into train and test
def train_test_split_time_series(X, y, split_ratio=0.8):
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

# Define models

# MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim, sequence_length):
        super(MLP, self).__init__()
        flattened_dim = input_dim * sequence_length
        self.layers = nn.Sequential(
            nn.Linear(flattened_dim, 64),  # Adjusted to flattened input
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.layers(x)

# RNN Model
class RNN(nn.Module):
    def __init__(self, input_dim):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, 32, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, 32, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# GRU Model
class GRU(nn.Module):
    def __init__(self, input_dim):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_dim, 32, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# CNN Model
class CNN(nn.Module):
    def __init__(self, input_dim, sequence_length):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear((sequence_length // 2) * 16, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch, channels, sequence_length)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# SVM Model
def train_svm(X_train, y_train):
    # Flatten the input to 2D
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    
    # Create and train the SVR model
    svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.5, gamma=1e-07))
    svr.fit(X_train_flattened, y_train)
    return svr

# Training Function
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


# Example Usage
# Load your dataset here
file_path = '../Input_Data/Processed_Files_Step1/Alphabet Inc.csv'
df = pd.read_csv(file_path)

df['Exchange Date'] = pd.to_datetime(df['Exchange Date'])
df.set_index('Exchange Date', inplace=True)
df = create_lagged_features(df, n_lags=3)

# Prepare features and target
# Preprocess data as before
features = [col for col in df.columns if col.startswith('Close_Lag') or col.startswith('Volume_Lag')]
X, y, scaler = preprocess_data(df[features + ['Close']], target_col='Close', sequence_length=3)

# Time series train-test split
X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # No shuffle for time series

# Testing DataLoader (if needed)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train MLP Model
mlp_model = MLP(input_dim=X_train.shape[2], sequence_length=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
train_model(mlp_model, train_loader, criterion, optimizer)

svr_model = train_svm(X_train, y_train)

# Function to make predictions and inverse transform the results
def predict_and_inverse_transform(model, X, scaler, feature_dim=1):
    """
    Predicts using the trained model and transforms predictions back to the original scale.

    Parameters:
    - model: Trained model (MLP or SVR).
    - X: Input features.
    - scaler: MinMaxScaler fitted during preprocessing.
    - feature_dim: Dimensionality of input features for reshaping.

    Returns:
    - Original scale predictions
    """
    if isinstance(model, nn.Module):  # MLP Model
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(X, dtype=torch.float32)).numpy()
    else:  # SVR Model
        X_flattened = X.reshape(X.shape[0], -1)  # Flatten input for SVR
        predictions = model.predict(X_flattened)
    
    # Inverse transform to original scale
    predictions_reshaped = predictions.reshape(-1, 1)  # For scaler compatibility
    dummy_features = np.zeros((len(predictions), feature_dim))  # Fill other features with zeros
    original_scale = scaler.inverse_transform(np.hstack([dummy_features, predictions_reshaped]))
    return original_scale[:, -1]  # Return the inverse-transformed predictions

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to evaluate predictions
def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    return mse, mae, r2

# Function to visualize predictions
def visualize_predictions(y_true, y_pred, title="Prediction Results"):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Actual", marker='o')
    plt.plot(y_pred, label="Predicted", marker='x')
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid()
    plt.show()

# Evaluate and visualize MLP predictions
mlp_predictions = predict_and_inverse_transform(mlp_model, X_test, scaler, feature_dim=X_test.shape[2])
y_test_original = scaler.inverse_transform(np.hstack([np.zeros((len(y_test), X_test.shape[2])), y_test.reshape(-1, 1)]))[:, -1]

print("Evaluation for MLP Predictions:")
mse, mae, r2 = evaluate_predictions(y_test_original, mlp_predictions)
visualize_predictions(y_test_original, mlp_predictions, title="MLP Prediction Results")

# Evaluate and visualize SVR predictions
svr_predictions = predict_and_inverse_transform(svr_model, X_test, scaler, feature_dim=X_test.shape[2])

print("Evaluation for SVR Predictions:")
mse, mae, r2 = evaluate_predictions(y_test_original, svr_predictions)
visualize_predictions(y_test_original, svr_predictions, title="SVR Prediction Results")

# Train and Predict for RNN, LSTM, GRU, and CNN
models = {
    "RNN": RNN(input_dim=X_train.shape[2]),
    "LSTM": LSTM(input_dim=X_train.shape[2]),
    "GRU": GRU(input_dim=X_train.shape[2]),
    "CNN": CNN(input_dim=X_train.shape[2], sequence_length=X_train.shape[1]),
}

# Training Loop for All Models
trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name} Model:")
    model = model.to(torch.device("cpu"))  # Ensure CPU usage
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_model(model, train_loader, criterion, optimizer)
    trained_models[name] = model

# Predict and Evaluate All Models
for name, model in trained_models.items():
    print(f"\nEvaluating {name} Model:")
    predictions = predict_and_inverse_transform(model, X_test, scaler, feature_dim=X_test.shape[2])
    evaluate_predictions(y_test_original, predictions)
    visualize_predictions(y_test_original, predictions, title=f"{name} Prediction Results")
