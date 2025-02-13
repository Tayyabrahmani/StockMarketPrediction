from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from statsmodels.stats.stattools import durbin_watson
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import pandas as pd
import shap
import argparse
from pathlib import Path

COUNTRY_STOCK_MAP = {
    "USA": ["NASDAQ 100 Index", "Alphabet Inc", "Berkshire Hathaway Inc", "Pfizer Inc"],
    "Germany": ["Deutsche Boerse DAX Index", "SAP SE", "Siemens AG", "Volkswagen AG"],
    "India": ["Nifty 50 Index", "Infosys Ltd", "Reliance Industries Ltd", "Tata Motors Ltd"]
}

ASSET_STOCK_MAP = {
    "Equities": ["Alphabet Inc", 'Berkshire Hathaway Inc',  'Deutsche Boerse DAX Index',
                 'Infosys Ltd', 'NASDAQ 100 Index', 'Nifty 50 Index', 'Pfizer Inc', 'Reliance Industries Ltd',
                 'SAP SE', 'Siemens AG', 'Tata Motors Ltd', 'Volkswagen AG'],
    "Commodities": ['CBoT Wheat Composite Commodity Future Continuation 1',  'Gold', 'ICE Europe Brent Crude Electronic Energy Future'],
    "Forex": ['Euro-Indian Rupee FX Cross Rate']
}

model_order = [
    "ARIMA", "XGBoost", "ARIMA-XGB", "CNN", "LSTM", 
    "RNN", "Transformers", "Crossformers", "PatchTST"
]

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

    # Calculate Mean Directional Accuracy (MDA)
    actual_direction = np.sign(y_true.diff().fillna(0))
    predicted_direction = np.sign(y_pred.diff().fillna(0))
    mda = (actual_direction == predicted_direction).mean() * 100

    return {
        # "MAE": mae,
        "RMSE": rmse,
        # "R²": r2,
        "SMAPE": smape,
        "EVS": evs,
        "Durbin-Watson": dw_stat,
        "MDA": mda,  # Add MDA to metrics
    }

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
        stock_metrics_df = stock_metrics_df.set_index("Model").reindex(model_order).reset_index()
        stock_metrics_df.to_csv(stock_metrics_path, index=False)
        print(f"Metrics for stock {stock} saved to {stock_metrics_path}")

def save_metrics_summary(input_dir, output_dir, model_column="Stock Name", stock_column="Model"):
    """
    Loads all CSV files from a directory, concatenates them into a single DataFrame,
    groups the data by metrics, unstacks at the stock level, and saves separate CSVs
    for each metric.

    Parameters:
        input_dir (str): Directory containing CSV files.
        output_dir (str): Directory to save processed metric-level CSVs.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dataframes = []

    # Load and concatenate all CSV files
    for file in input_dir.glob("*_metrics.csv"):
        try:
            df = pd.read_csv(file)
            df['Stock Name'] = file.name.split("_")[0]
            all_dataframes.append(df)
        except Exception as e:
            print(f"❌ Error loading {file.name}: {e}")

    if not all_dataframes:
        print("❌ No valid CSV files found.")
        return

    # Concatenate all data into a single DataFrame
    full_df = pd.concat(all_dataframes, ignore_index=True)

    # Reshape the data: Convert metric names into rows & stock names into columns
    melted_df = full_df.melt(id_vars=[model_column, stock_column], var_name="Metric", value_name="Value")
    
    for metric, metric_df in melted_df.groupby("Metric"):
        # Pivot: Model -> Index, Stock Name -> Columns
        pivot_df = metric_df.pivot(index=stock_column, columns=model_column, values="Value")
        pivot_df = pivot_df.reindex(index=model_order)

        # Save the transposed metric file
        pivot_df.to_csv(output_dir / f"{metric}_summary.csv")

def save_country_metrics_summary(input_dir, output_dir, model_column="Stock Name", stock_column="Model", summary_type="Country"):
    """
    Loads all CSV files from a directory, concatenates them into a single DataFrame,
    groups the data by country based on the provided mapping, and saves separate CSVs
    for each country.

    Parameters:
        input_dir (str): Directory containing CSV files.
        output_dir (str): Directory to save processed metric-level CSVs.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dataframes = []

    # Load and concatenate all CSV files
    for file in input_dir.glob("*_metrics.csv"):
        try:
            df = pd.read_csv(file)
            df['Stock Name'] = file.name.split("_")[0]
            all_dataframes.append(df)
        except Exception as e:
            print(f"❌ Error loading {file.name}: {e}")

    if not all_dataframes:
        print("❌ No valid CSV files found.")
        return

    # Concatenate all data into a single DataFrame
    full_df = pd.concat(all_dataframes, ignore_index=True)

    # Reshape the data: Convert metric names into rows & stock names into columns
    melted_df = full_df.melt(id_vars=[model_column, stock_column], var_name="Metric", value_name="Value")

    dict_type = {"Country": COUNTRY_STOCK_MAP, "Asset": ASSET_STOCK_MAP}
    dict_type = dict_type[summary_type]

    for country, stocks in dict_type.items():
        country_df = melted_df[melted_df[model_column].isin(stocks)]
        
        if country_df.empty:
            print(f"⚠️ No data found for {country}, skipping...")
            continue

        for metric, metric_df in country_df.groupby("Metric"):
            # Pivot: Model -> Index, Stock Name -> Columns
            pivot_df = metric_df.pivot(index=stock_column, columns=model_column, values="Value")
            pivot_df = pivot_df.reindex(index=model_order)

            # Save country-specific metric file
            country_output_dir = output_dir / country
            country_output_dir.mkdir(parents=True, exist_ok=True)
            pivot_df.to_csv(country_output_dir / f"{metric}_summary.csv")

            print(f"✅ Saved {metric}_summary.csv for {country}")

def process_and_plot_metrics(input_dir, output_dir):
    """
    Loads metric summary files, adds an average column across models, 
    creates bar charts for each metric, and saves results.

    Parameters:
        input_dir (str): Directory containing metric CSV files.
        output_dir (str): Directory to save processed CSVs and charts.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = list(input_dir.glob("*.csv"))

    if not all_files:
        print("❌ No metric summary files found.")
        return

    for file in all_files:
        try:
            # Load CSV
            df = pd.read_csv(file)
            df = df.set_index("Model").reindex(model_order).reset_index()

            # Calculate average across models (ignoring first column which is 'Model')
            df["Average"] = df.iloc[:, 1:].mean(axis=1)
            df = df[["Model", "Average"]]

            # Generate bar chart
            plt.figure(figsize=(12, 6))
            df.set_index("Model").plot(kind="bar", figsize=(12, 6))
            plt.title(f"{file.stem.split('_')[0]} - Model Comparison")
            plt.ylabel("Metric Value")
            plt.xticks(rotation=45)
            plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')

            # Save chart
            plt.tight_layout()
            plt.savefig(output_dir / f"{file.stem.split('_')[0]}_chart.png")
            plt.close()

            print(f"✅ Processed & saved: {file.name}")

        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")

def process_and_plot_metrics_region(input_dir: str, output_dir: str, regions=["USA", "Germany", "India"]):
    """
    Reads RMSE values from RMSE_Summary.csv files in each region folder,
    calculates average RMSE per forecasting model, and generates a grouped bar chart.

    Args:
        input_dir (str): Path to the folder containing regional RMSE_Summary.csv files.
        output_dir (str): Path to save the generated bar chart.
    """
    rmse_data = {}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read SMAPE values from SMAPE_Summary.csv files
    for region in regions:
        file_path = os.path.join(input_dir, region, "SMAPE_Summary.csv")

        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist. Skipping {region}.")
            continue

        try:
            # Read CSV file (first column is Model names, rest are SMAPE values for different stocks)
            df = pd.read_csv(file_path)

            # Ensure first column is "Model" and extract model names
            df.rename(columns={df.columns[0]: "Model"}, inplace=True)

            # Compute the average SMAPE across all stock columns (excluding "Model" column)
            df["Average_SMAPE"] = df.iloc[:, 1:].mean(axis=1)

            # Store SMAPE data in a dictionary
            for _, row in df.iterrows():
                model = row["Model"]
                rmse_value = row["Average_SMAPE"]
                if model not in rmse_data:
                    rmse_data[model] = {}
                rmse_data[model][region] = rmse_value

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Convert to DataFrame
    df_final = pd.DataFrame.from_dict(rmse_data, orient="index").fillna(0)

    # Plot grouped bar chart
    models = df_final.index
    x = np.arange(len(models))
    width = 0.25  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for each region
    colors = {regions[0]: "blue", regions[1]: "red", regions[2]: "green"}

    # Plot bars for each region
    for i, region in enumerate(regions):
        ax.bar(x + i * width, df_final[region], width, label=region, color=colors.get(region, "gray"))

    # Formatting
    ax.set_xlabel("Forecasting Models")
    ax.set_ylabel("Average SMAPE")
    ax.set_title("SMAPE Comparison Across Forecasting Models")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()

    # Save the figure
    output_path = os.path.join(output_dir, "SMAPE_chart.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Chart saved at: {output_path}")

def save_metric_bar_charts(metrics_df, output_dir="Output_Data"):
    """
    Saves RMSE bar charts for each stock at the stock level.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics for all models.
        output_dir (str): Directory to save the bar charts.
    """
    metrics = ["RMSE", "SMAPE", "EVS", "Durbin-Watson", "MDA"]

    for metric in metrics:
        metric_chart_dir = os.path.join(output_dir, "charts", f"{metric.lower()}_by_stock")
        os.makedirs(metric_chart_dir, exist_ok=True)

        for stock in metrics_df["Stock"].unique():
            stock_df = metrics_df[metrics_df["Stock"] == stock]
            stock_df = stock_df.sort_values(by="Model")

            plt.figure(figsize=(10, 6))
            plt.bar(stock_df["Model"], stock_df[metric], color="skyblue")

            plt.xlabel("Models")
            plt.ylabel(metric)
            plt.title(f"{metric} Comparison for {stock}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            chart_path = os.path.join(metric_chart_dir, f"{stock}_{metric.lower()}_chart.png")
            plt.savefig(chart_path, dpi=300)
            plt.close()

def reorder_metrics_table(metrics_df):
    """
    Reorders the metrics table based on the given model order.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics for all models.

    Returns:
        pd.DataFrame: Reordered DataFrame.
    """
    metrics_df["Model"] = pd.Categorical(metrics_df["Model"], categories=model_order, ordered=True)
    return metrics_df.sort_values(by=["Stock", "Model"])

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
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate models and save metrics.")
    parser.add_argument(
        "--filter",
        type=str,
        default="Alphabet Inc",
        help="Filter by a specific model or stock name (optional).",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Define paths
    predictions_dir = "Output_Data/saved_predictions"
    actual_data_dir = "Input_Data/Processed_Files_Step1"

    # Evaluate models and generate metrics table
    metrics_df, stock_metrics = evaluate_models(predictions_dir, actual_data_dir)

    # Reorder metrics table
    metrics_df = reorder_metrics_table(metrics_df)

    # Save metrics table
    save_metrics_table(metrics_df, stock_metrics)

    # Save Metrics level summary for all the metrics
    save_metrics_summary(input_dir="Output_Data/saved_metrics", output_dir="Output_Data/saved_metrics/overall_summary")

    # Save Metrics level summary country level
    save_country_metrics_summary(input_dir="Output_Data/saved_metrics", output_dir="Output_Data/saved_metrics/region_summary", summary_type="Country")

    # Save Metrics level summary asset level
    save_country_metrics_summary(input_dir="Output_Data/saved_metrics", output_dir="Output_Data/saved_metrics/asset_summary", summary_type="Asset")

    # Save RMSE bar chart
    save_metric_bar_charts(metrics_df, output_dir="Output_Data/saved_metrics")

    # Save bar chart for overall summary
    process_and_plot_metrics(input_dir="Output_Data/saved_metrics/overall_summary", output_dir="Output_Data/charts/overall_summary")

    # Save bar chart for region summary
    process_and_plot_metrics_region(input_dir="Output_Data/saved_metrics/region_summary", output_dir="Output_Data/charts/region_summary")

    # Save bar chart for Asset summary
    process_and_plot_metrics_region(input_dir="Output_Data/saved_metrics/asset_summary", output_dir="Output_Data/charts/asset_summary", regions=["Equities", "Commodities", "Forex"])

    print("\nEvaluation Metrics for All Models:")

    filtered_metrics = metrics_df[metrics_df["Stock"] == args.filter]
    print(filtered_metrics)
