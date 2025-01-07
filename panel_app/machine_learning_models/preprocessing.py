import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
lags = [1, 2, 3, 7, 14, 28]

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
        
        # Remove columns which have similar value as the target variable
        data = data.drop(['Open', 'Low', 'High', 'Volume', 'Flow', 'Turnover - USD', 'Stock Name'], axis=1)
        return data
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")

def create_lagged_features(df, target_col='Close', lags=lags, rolling_window=None):
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
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    if rolling_window:
        df[f"{target_col}_roll_mean"] = df[target_col].rolling(window=rolling_window).mean()
        df[f"{target_col}_roll_std"] = df[target_col].rolling(window=rolling_window).std()
    return df

def extract_date_features(df, date_column='Exchange Date'):
    """
    Extracts date-based features from a datetime column.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        date_column (str): Name of the datetime column.
    
    Returns:
        pd.DataFrame: DataFrame with additional date-based features.
    """
    df = df.reset_index()
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in the dataframe.")
    
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    if df[date_column].isnull().any():
        raise ValueError(f"Date column '{date_column}' contains invalid datetime entries.")
    
    # Extract features
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['Day'] = df[date_column].dt.day
    df['DayOfWeek'] = df[date_column].dt.dayofweek
    df['WeekOfYear'] = df[date_column].dt.isocalendar().week
    df['Quarter'] = df[date_column].dt.quarter
    df['IsMonthStart'] = df[date_column].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df[date_column].dt.is_month_end.astype(int)
    df['IsWeekend'] = (df[date_column].dt.dayofweek >= 5).astype(int)
    
    df = df.set_index('Exchange Date')
    return df

def fill_na_values(df):
    """
    Fills NaN values in the dataframe using forward fill (ffill).

    Parameters:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        pd.DataFrame: DataFrame with NaN values filled using forward fill.
    """
    return df.ffill().bfill()

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
    df = df[[col for col in df.columns if col != target_col] + [target_col]]

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

    # Extract the target column as a Series
    target_series = df[target_col]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exog_cols = numeric_cols.drop([target_col], errors="ignore")
    if len(exog_cols) == 0:
        raise ValueError("No exogenous numeric columns found for preprocessing.")

    exogenous = df[exog_cols]

    # Scale exogenous variables
    scaler = MinMaxScaler()
    exogenous_scaled = scaler.fit_transform(exogenous)
    exogenous_scaled_df  = pd.DataFrame(exogenous_scaled, columns=exog_cols)
    return exogenous_scaled_df, target_series, scaler

def train_test_split_time_series(X, y, test_size=0.02):
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
