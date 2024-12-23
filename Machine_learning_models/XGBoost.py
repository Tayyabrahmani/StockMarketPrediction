import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import os
import pickle

class XGBoostStockModel:
    def __init__(self, file_path, stock_name, lags=3, rolling_window=3, test_size=0.2, hyperparameters=None):
        """
        Initializes the XGBoost model for stock prediction.

        Parameters:
            file_path (str): Path to the CSV file containing stock data.
            stock_name (str): Name of the stock for saving models and predictions.
            lags (int): Number of lag features to create.
            rolling_window (int): Size of the rolling window for feature engineering.
            test_size (float): Fraction of the data to use for testing.
            hyperparameters (dict): Hyperparameters for the XGBoost model.
        """
        self.file_path = file_path
        self.stock_name = stock_name
        self.lags = lags
        self.rolling_window = rolling_window
        self.test_size = test_size
        self.hyperparameters = hyperparameters or {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42
        }
        
        self.data = self.load_data()
        self.features, self.target = self.create_features_and_target()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.model = None

    def load_data(self):
        """
        Loads and preprocesses the stock data.

        Returns:
            pd.DataFrame: The preprocessed stock data.
        """
        data = pd.read_csv(self.file_path)
        data['Exchange Date'] = pd.to_datetime(data['Exchange Date'])
        data.set_index('Exchange Date', inplace=True)
        return data

    def create_features_and_target(self):
        """
        Creates lag and rolling window features for the data and ensures all features are numeric.

        Returns:
            tuple: (features, target) DataFrames.
        """
        data = self.data.copy()
        
        # Create lag features
        for lag in range(1, self.lags + 1):
            data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
        
        # Create rolling window features
        data[f'Close_roll_mean_{self.rolling_window}'] = data['Close'].rolling(window=self.rolling_window).mean()
        data[f'Close_roll_std_{self.rolling_window}'] = data['Close'].rolling(window=self.rolling_window).std()

        # Drop rows with NaN values
        data.dropna(inplace=True)

        # Ensure only numeric features
        features = data.drop(columns=['Close'], errors='ignore')
        features = features.select_dtypes(include=['number'])

        # Target variable
        target = data['Close']
        return features, target

    def split_data(self):
        """
        Splits the data into train and test sets.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        split_index = int(len(self.features) * (1 - self.test_size))
        X_train = self.features.iloc[:split_index]
        X_test = self.features.iloc[split_index:]
        y_train = self.target.iloc[:split_index]
        y_test = self.target.iloc[split_index:]
        return X_train, X_test, y_train, y_test

    def train(self):
        """
        Trains the XGBoost model.

        Returns:
            xgb.XGBRegressor: The trained model.
        """
        self.model = xgb.XGBRegressor(**self.hyperparameters)
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self):
        """
        Evaluates the model's performance.

        Returns:
            dict: Evaluation metrics (MAE, RMSE, R2).
        """
        predictions = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2 = r2_score(self.y_test, predictions)
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    def save_model(self):
        """
        Saves the trained model as a pickle file.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_xgboost_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def save_predictions(self):
        """
        Saves the predictions as a CSV file.
        """
        predictions = self.model.predict(self.X_test)
        forecast_df = pd.DataFrame({
            "Date": self.X_test.index,
            "Predicted Close": predictions
        })
        prediction_dir = "Output_Data/saved_predictions"
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, f"XGBoost_{self.stock_name}_forecast.csv")
        forecast_df.to_csv(prediction_path, index=False)

    def run(self):
        """
        Runs the full pipeline: trains the model, evaluates it, saves the model and predictions.
        """
        print(f"Training XGBoost model for {self.stock_name}...")
        self.train()
        metrics = self.evaluate()
        print(f"Evaluation Metrics for {self.stock_name}: {metrics}")
        self.save_model()
        self.save_predictions()
        return metrics

if __name__ == "__main__":
    # Initialize the model
    model = XGBoostStockModel(
        file_path='Input_Data/Processed_Files_Step1/Alphabet Inc.csv',
        stock_name='Alphabet Inc',
        lags=5,
        rolling_window=3,
        test_size=0.2,
        hyperparameters={
            "objective": "reg:squarederror",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": 42
        }
    )

    # Run the full pipeline
    metrics = model.run()
    print(f"Final Metrics: {metrics}")