from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def evaluate_predictions(y_true, y_pred):
    """
    Calculates evaluation metrics.

    Parameters:
        y_true (np.array): Ground truth values.
        y_pred (np.array): Predicted values.
    
    Returns:
        dict: Metrics (MAE, RMSE, R²).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R²": r2}


def visualize_predictions(y_true, y_pred, title="Prediction Results"):
    """
    Visualizes predictions against ground truth.

    Parameters:
        y_true (np.array): Ground truth values.
        y_pred (np.array): Predicted values.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Actual", marker='o')
    plt.plot(y_pred, label="Predicted", marker='x')
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid()
    plt.show()


def predict_and_inverse_transform(model, X, scaler, feature_dim):
    """
    Predicts using the trained model and transforms predictions back to the original scale.

    Parameters:
        model: Trained model (MLP, RNN, etc.).
        X (np.array): Input features.
        scaler: Fitted scaler.
        feature_dim (int): Dimensionality of input features (excluding the target).
    
    Returns:
        np.array: Inverse transformed predictions.
    """
    if hasattr(model, 'predict'):  # For sklearn models like SVR
        predictions = model.predict(X.reshape(X.shape[0], -1))
    else:  # For PyTorch models
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(X, dtype=torch.float32)).numpy()
    
    # Prepare for inverse transformation
    predictions_reshaped = predictions.reshape(-1, 1)  # For scaler compatibility
    dummy_features = np.zeros((len(predictions), feature_dim))  # Match scaler dimensions
    original_scale = scaler.inverse_transform(
        np.hstack([dummy_features, predictions_reshaped])
    )
    return original_scale[:, -1]
