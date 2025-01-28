import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import EfficientFCParameters
from tsfresh.feature_selection.significance_tests import target_real_feature_real_test


def add_tsfresh_features(df, n_top_features=16):
    """
    Adds TSFresh features to the stock data.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock data with 'Exchange Date' and 'Close'.
        n_top_features (int): Number of top features to select based on significance.

    Returns:
        pd.DataFrame: DataFrame with TSFresh features added.
    """
    # Generate the indices for repeated rows
    repeated_indices = np.concatenate([np.arange(i + 1) for i in range(len(df))])

    # Create the expanded DataFrame
    expanded_df = df.iloc[repeated_indices].reset_index(drop=True)

    # Generate the `id` column (group indicator)
    id_column = np.concatenate([np.full(i + 1, i) for i in range(len(df))])
    expanded_df['id'] = id_column

    # Extract TSFresh features
    extracted_features = extract_features(
        expanded_df,
        column_id="id",
        column_sort="Exchange Date",
        column_value="Close",
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=14  # Adjust based on available resources
    )

    # Handle NaN values
    extracted_features = extracted_features.fillna(0)

    # Compute p-values for each feature
    p_values = {}
    for feature in extracted_features.columns:
        feature_series = extracted_features[feature]
        p_value = target_real_feature_real_test(feature_series, df["Close"])
        p_values[feature] = p_value

    # Select the top N features based on p-values
    p_values_series = pd.Series(p_values)
    top_features = p_values_series.nsmallest(n_top_features).index
    selected_features = extracted_features[top_features]

    # Reset the index and prepare for merging
    selected_features = selected_features.reset_index()
    selected_features = selected_features.rename(columns={"index": "id"})
    selected_features['id'] = selected_features['id'] + 1

    # Merge the selected features with the original data
    df = df.reset_index()
    df = df.rename(columns={"index": "id"})
    final_df = df.merge(selected_features, on='id', how='left')

    # Drop unnecessary columns
    final_df = final_df.drop(['id'], axis=1)

    if 'Close__length' in final_df.columns:
        final_df = final_df.drop(['Close__length'], axis=1)

    return final_df


def process_all_stocks_with_tsfresh(input_dir, output_dir, n_top_features=10):
    """
    Processes all stock CSV files in the input directory, adds TSFresh features,
    and saves the enhanced datasets to the output directory.

    Parameters:
        input_dir (str): Path to the directory containing input CSV files.
        output_dir (str): Path to the directory to save enhanced CSV files.
        n_top_features (int): Number of top TSFresh features to select.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each CSV file in the input directory
    for stock_file in input_path.glob("*.csv"):
        if 'Alphabet Inc' not  in stock_file.name:
            pass
        try:
            print(f"Processing file: {stock_file.name}")

            # Load stock data
            df = pd.read_csv(stock_file)

            # Ensure required columns are present
            required_columns = ["Exchange Date", "Close"]
            if not all(col in df.columns for col in required_columns):
                print(f"Skipping {stock_file.name}: Missing required columns")
                continue

            # Sort data by date
            df["Exchange Date"] = pd.to_datetime(df["Exchange Date"])
            df = df.sort_values("Exchange Date")

            # Add TSFresh features
            df = add_tsfresh_features(df, n_top_features=n_top_features)

            # Save enhanced data
            output_file = output_path / stock_file.name
            df.to_csv(output_file, index=False)
            print(f"Saved enhanced data to: {output_file}")

        except Exception as e:
            print(traceback.format_exc())
            print(f"Error processing file {stock_file.name}: {e}")


# Example usage
if __name__ == "__main__":
    input_dir = "Input_Data/Processed_Files_Step2"
    output_dir = "Input_Data/Processed_Files_Step3"
    process_all_stocks_with_tsfresh(input_dir, output_dir)
