from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from statsmodels.stats.stattools import durbin_watson
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import pandas as pd
import shap

def plot_shap_feature_importance(model, X_train, feature_names, stock_name):
    """
    Plots and saves the SHAP feature importance based on SHAP values.

    Parameters:
        model: Trained model (e.g., SVR).
        X_train (np.array): Training features (scaled).
        feature_names (list or pd.Index): Names of the features.
        stock_name (str): Name of the stock for labeling the plot.

    Returns:
        None
    """

    random_indices = np.random.choice(X_train.shape[0], 50, replace=False)
    X_train = X_train[random_indices]

    # Ensure SHAP supports the model type
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train)

    # Generate summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", show=False)

    # Customize the plot title
    plt.title(f"SHAP Feature Importance for {stock_name}")
    plt.tight_layout()

    # Save the plot
    output_dir = "Output_Data/saved_feature_importance"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"SHAP_{stock_name}_feature_importance.png")
    plt.savefig(plot_path, dpi=300)

    # Show the plot
    plt.show()

    print(f"SHAP feature importance plot saved to: {plot_path}")

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
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    evs = explained_variance_score(y_true, y_pred)
    residuals = y_true - y_pred
    dw_stat = durbin_watson(residuals) 
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "SMAPE": smape, "EVS": evs, "Durbin-Watson": dw_stat}

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
    stock_metrics = {}

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
        metrics_dict = evaluate_predictions(y_true, y_pred)      
        metrics_list.append({
            "Model": model_name,
            "Stock": stock_name,
            **metrics_dict,
        })

        # Collect metrics by stock
        if stock_name not in stock_metrics:
            stock_metrics[stock_name] = []
        stock_metrics[stock_name].append({"Model": model_name, **metrics_dict})

    metrics_table = pd.DataFrame(metrics_list)
    metrics_table = metrics_table.round(3)
    metrics_table = metrics_table.sort_values(["Stock", "Model"])
    return metrics_table, stock_metrics

def save_metrics_table(metrics_df, stock_metrics, output_dir="Output_Data"):
    """
    Saves the metrics table to a CSV file.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics for all models.
        output_dir (str): Directory to save the metrics table.
    """
    overall_metrics_dir = os.path.join(output_dir, "saved_metrics")
    os.makedirs(overall_metrics_dir, exist_ok=True)

    overall_metrics_path = os.path.join(output_dir, "model_metrics.csv")
    metrics_df.to_csv(overall_metrics_path, index=False)
    print(f"Overall metrics table saved to {overall_metrics_path}")

    # Save individual stock metrics
    for stock, metrics in stock_metrics.items():
        stock_metrics_path = os.path.join(overall_metrics_dir, f"{stock}_metrics.csv")
        stock_metrics_df = pd.DataFrame(metrics).round(3)
        stock_metrics_df.to_csv(stock_metrics_path, index=False)
        print(f"Metrics for stock {stock} saved to {stock_metrics_path}")

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
    actual_data_dir = "Input_Data/Processed_Files_Step1"

    # Evaluate models and generate metrics table
    metrics_df, stock_metrics = evaluate_models(predictions_dir, actual_data_dir)

    # Save metrics table
    save_metrics_table(metrics_df, stock_metrics)

    print("\nEvaluation Metrics for All Models:")
    print(metrics_df)
