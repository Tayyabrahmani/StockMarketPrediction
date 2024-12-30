from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import pandas as pd

def evaluate_predictions(y_true, y_pred):
    """
    Calculates evaluation metrics.

    Parameters:
        y_true (pd.Series): Ground truth values.
        y_pred (pd.Series): Predicted values.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R²": r2}

def evaluate_models(predictions_dir, actual_data_dir):
    """
    Evaluates all models using their predictions and ground truth values for each stock.

    Parameters:
        predictions_dir (str): Directory containing model predictions CSVs.
        actual_data_dir (str): Directory containing actual stock price CSVs.

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for all models.
    """
    metrics_list = []

    # Iterate over all prediction files
    for pred_file in os.listdir(predictions_dir):
        if not pred_file.endswith(".csv"):
            continue
        
        # Extract stock name and model name from the prediction file name
        parts = pred_file.split("_")
        stock_name = "_".join(parts[1:-1])  # Assume the format is ModelName_StockName_predictions.csv
        model_name = parts[0]

        pred_file_path = os.path.join(predictions_dir, pred_file)
        
        # Load predictions
        predictions_df = pd.read_csv(pred_file_path)

        # Find the corresponding ground truth file
        actual_file_path = os.path.join(actual_data_dir, f"{stock_name}.csv")
        if not os.path.exists(actual_file_path):
            print(f"Ground truth file not found for stock: {stock_name}")
            continue
        
        actual_df = pd.read_csv(actual_file_path)
        actual_df['Exchange Date'] = pd.to_datetime(actual_df['Exchange Date'])
        actual_df.set_index('Exchange Date', inplace=True)

        # Align ground truth with predictions
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        merged_df = pd.merge(predictions_df, actual_df, left_on="Date", right_index=True, how="inner")
        if merged_df.empty:
            print(f"No overlapping dates found for stock: {stock_name}")
            continue

        y_true = merged_df['Close']
        y_pred = merged_df['Predicted Close']

        # Compute metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        metrics_list.append({
            "Model": model_name,
            "Stock": stock_name,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
        })

    metrics_table = pd.DataFrame(metrics_list)
    metrics_table = metrics_table.round(3)
    return metrics_table

def save_metrics_table(metrics_df, output_dir="Output_Data"):
    """
    Saves the metrics table to a CSV file.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics for all models.
        output_dir (str): Directory to save the metrics table.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics table saved to {metrics_path}")

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

if __name__ == "__main__":
    # Define paths
    predictions_dir = "Output_Data/saved_predictions"
    actual_data_dir = "Input_Data/Processed_Files_Step2"

    # Evaluate models and generate metrics table
    metrics_df = evaluate_models(predictions_dir, actual_data_dir)

    # Save metrics table
    save_metrics_table(metrics_df)

    print("\nEvaluation Metrics for All Models:")
    print(metrics_df)
