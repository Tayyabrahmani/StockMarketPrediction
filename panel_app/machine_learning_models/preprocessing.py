import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Loads stock data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded and preprocessed data.
    """
    try:
        data = pd.read_csv(file_path)
        data['Exchange Date'] = pd.to_datetime(data['Exchange Date'])
        data.set_index('Exchange Date', inplace=True)
        if 'Close' not in data.columns:
            raise ValueError(f"'Close' column is missing in the input data: {file_path}")
        return data[['Close']]  # Only return the 'Close' column
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")

def create_lagged_features(df, target_col='Close', lags=3, rolling_window=None):
    """
    Creates lagged and rolling window features for time-series data.

    Parameters:
        df (pd.DataFrame): The input data.
        target_col (str): The target column name.
        lags (int): Number of lag features to create.
        rolling_window (int, optional): Size of the rolling window for features.
    
    Returns:
        pd.DataFrame: Data with lagged and rolling window features.
    """
    for lag in range(1, lags + 1):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    if rolling_window:
        df[f"{target_col}_roll_mean"] = df[target_col].rolling(window=rolling_window).mean()
        df[f"{target_col}_roll_std"] = df[target_col].rolling(window=rolling_window).std()
    df.dropna(inplace=True)
    return df


def preprocess_data(df, target_col='Close', sequence_length=30):
    """
    Prepares data for training by creating sequences and scaling.

    Parameters:
        df (pd.DataFrame): The input data.
        target_col (str): The target column name.
        sequence_length (int): Number of time steps for sequences.
    
    Returns:
        tuple: Scaled features (X), target (y), and fitted scaler.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length, :-1])
        y.append(scaled_data[i + sequence_length, -1])
    return np.array(X), np.array(y), scaler

def preprocess_data_for_arima(df, target_col='Close'):
    """
    Prepares raw data for ARIMA by ensuring proper indexing and target column.

    Parameters:
        df (pd.DataFrame): The input data.
        target_col (str): The target column name.
    
    Returns:
        pd.Series: The time-series data for ARIMA.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataframe.")
    return df[target_col]

def train_test_split_time_series(X, y, test_size=0.05):
    """
    Splits the data into training and testing sets.

    Parameters:
        X (np.array): Features.
        y (np.array): Target variable.
        test_size (float): Fraction of the data to use for testing.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if len(X) != len(y):
        raise ValueError("Features (X) and target (y) must have the same length.")
    if len(X) == 0:
        raise ValueError("Input data (X) is empty.")
    
    split_idx = int(len(X) * (1 - test_size))
    if split_idx == 0 or split_idx == len(X):
        raise ValueError("Test size too small or too large for the data.")
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
