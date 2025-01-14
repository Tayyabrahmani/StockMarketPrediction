import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

def preprocess_data_svr(X_train, X_test, y_train, y_test, X_val, y_val):
    """
    Prepares data for SVR by scaling both features and target values.

    Parameters:
        X_train (np.array or pd.DataFrame): Training features (2D).
        X_test (np.array or pd.DataFrame): Test features (2D).
        y_train (np.array or pd.Series): Training target values (1D).
        y_test (np.array or pd.Series): Test target values (1D).

    Returns:
        tuple: Scaled features (X_train_scaled, X_test_scaled), scaled targets (y_train_scaled, y_test_scaled),
               feature scaler, and target scaler.
    """
    # Convert to NumPy arrays if necessary
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    X_val = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
     
    y_train = y_train.values if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train
    y_test = y_test.values if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test
    y_val = y_val.values if isinstance(y_val, (pd.Series, pd.DataFrame)) else y_val

    # Feature scaling
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    # Target scaling
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_val_scaled, y_val_scaled, feature_scaler, target_scaler

def preprocess_data(X_train, X_test, X_val, y_train, y_test, y_val, add_feature_dim=False):
    """
    Prepares data for training by scaling features and targets.

    Parameters:
        X_train, X_test: Training and testing feature datasets.
        y_train, y_test: Training and testing target datasets.
        add_feature_dim (bool): If True, adds a feature dimension for models like transformers.

    Returns:
        tuple: Scaled features (X_train_scaled, X_test_scaled), 
               targets (y_train_scaled, y_test_scaled), 
               and the fitted scalers (feature_scaler, target_scaler).
    """
    # Initialize scalers
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Convert to NumPy arrays
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    X_val = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
    y_train = y_train.values if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train
    y_test = y_test.values if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test
    y_val = y_val.values if isinstance(y_val, (pd.Series, pd.DataFrame)) else y_val

    if add_feature_dim:
        # Feature scaling
        n_samples, sequence_length, n_features = X_train.shape
        X_train = X_train.reshape(-1, n_features)

        n_samples_test, sequence_length_test, n_features_test = X_test.shape
        X_test = X_test.reshape(-1, n_features)

        n_samples_val, sequence_length_val, n_features_val = X_val.shape
        X_val = X_val.reshape(-1, n_features)

    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    X_val_scaled = feature_scaler.transform(X_val)

    if add_feature_dim:
        # Reshape back to original dimensions
        X_train_scaled = X_train_scaled.reshape(n_samples, sequence_length, n_features)
        X_test_scaled = X_test_scaled.reshape(n_samples_test, sequence_length_test, n_features_test)
        X_val_scaled = X_val_scaled.reshape(n_samples_val, sequence_length_val, n_features_val)

    # Target scaling
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

    return X_train_scaled, X_test_scaled, X_val_scaled, y_train_scaled, y_test_scaled, y_val_scaled, feature_scaler, target_scaler

def create_sequences(df, sequence_length=30, target_col="Close", is_df=True):
    """
    Creates sequences from the data for time series forecasting.

    Parameters:
        data (np.array): The scaled data.
        sequence_length (int): Number of time steps for sequences.
    
    Returns:
        tuple: Features (X) and target (y) sequences.
    """
    if is_df:
        df = df[[col for col in df.columns if col != target_col] + [target_col]].values

    X, y = [], []
    for i in range(len(df) - sequence_length):
        # X.append(df[i:i + sequence_length, :-1])
        X.append(df[i+1:i + sequence_length + 1, :-1])
        y.append(df[i + sequence_length, -1])
    return np.array(X), np.array(y)

def create_sequences_transformer(data, sequence_length, target_col):
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:i+sequence_length].drop(columns=[target_col]).values
        target = data.iloc[i+sequence_length][target_col]
        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

    print("Sequences shape:", sequences.shape)  # Debug
    print("Targets shape:", targets.shape)      # Debug

    return sequences, targets

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
