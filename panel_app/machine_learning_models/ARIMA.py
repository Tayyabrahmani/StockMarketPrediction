import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
import os
import pickle
from machine_learning_models.preprocessing import load_data, preprocess_data_for_arima, train_test_split_time_series, fill_na_values
from logging_config import get_logger

class ARIMAStockModel:
    def __init__(self, file_path, stock_name, max_p=5, max_q=5):
        """
        Initializes the ARIMA model for stock prediction.

        Parameters:
            file_path (str): Path to the CSV file containing stock data.
            stock_name (str): Name of the stock for saving models and predictions.
            forecast_period (int): Number of future periods to forecast.
            max_p (int): Maximum AR parameter for auto_arima.
            max_q (int): Maximum MA parameter for auto_arima.
        """
        self.logger = get_logger(__name__)  # Logger specific to this module
        self.logger.info(f"Initializing ARIMA model for {stock_name}...")

        self.file_path = file_path
        self.stock_name = stock_name
        self.max_p = max_p
        self.max_q = max_q
        self.model = None

        # Load data and handle technical indicators
        self.data = load_data(self.file_path)
        self.data = fill_na_values(self.data)

        # Use 'Close' as the target and technical indicators as exogenous variables
        self.exogenous, self.target, self.scaler = preprocess_data_for_arima(self.data, target_col="Close")

        # Split into train/test sets
        self.train_data, self.test_data, self.train_exog, self.test_exog = self._split_data()

    def _split_data(self):
        """
        Splits the data into training and testing sets, ensuring the `Close` column is preserved.
        """
        if "Close" not in self.data.columns:
            raise ValueError("The 'Close' column is missing in the dataset.")

        target_values = self.data["Close"].values
        train_indices, test_indices, train_values, test_values = train_test_split_time_series(
            self.data.index, target_values
        )

        train_exog = self.exogenous.iloc[:len(train_indices)].values if self.exogenous is not None else None
        test_exog = self.exogenous.iloc[len(train_indices):].values if self.exogenous is not None else None

        train_data = pd.DataFrame(train_values, index=train_indices, columns=["Close"])
        test_data = pd.DataFrame(test_values, index=test_indices, columns=["Close"])

        return train_data, test_data, train_exog, test_exog

    def test_stationarity(self, timeseries):
        """
        Tests the stationarity of the given time series using the Augmented Dickey-Fuller test.

        Parameters:
            timeseries (pd.Series): The time series to test.
        """
        adft = adfuller(timeseries.dropna(), autolag="AIC")
        print(f"Dickey-Fuller Test Results:\n"
              f"Test Statistic = {adft[0]:.4f}\n"
              f"p-value = {adft[1]:.4f}\n")
        return adft[1]

    def validate_model(self, n_splits=5):
        """
        Performs time-series cross-validation to validate the ARIMA model.

        Parameters:
            n_splits (int): Number of cross-validation splits.

        Returns:
            list: RMSE scores for each fold.
        """
        self.logger.info("Starting model validation using cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmse_scores = []

        # Iterate through each fold in cross-validation
        for fold_idx, (train_index, test_index) in enumerate(tscv.split(self.train_data)):
            self.logger.info(f"Processing fold {fold_idx + 1}/{n_splits}...")

            # Split the train and test sets for the current fold
            train_fold = self.train_data.iloc[train_index]
            test_fold = self.train_data.iloc[test_index]

            exog_train = self.train_exog[train_index] if self.train_exog is not None else None
            exog_test = self.train_exog[test_index] if self.train_exog is not None else None

            # Log-transform the training data
            train_fold_log = np.log(train_fold["Close"])

            # Train the ARIMA model on the current fold
            model = auto_arima(
                train_fold_log,
                exogenous=exog_train,
                start_p=0,
                start_q=0,
                max_p=self.max_p,
                max_q=self.max_q,
                d=self.differencing_order,
                seasonal=False,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
                stepwise=True,
            )

            # Predict on the test fold
            if exog_test is not None:
                predictions_log = model.predict(n_periods=len(test_fold), X=exog_test)
            else:
                predictions_log = model.predict(n_periods=len(test_fold))

            # Transform predictions back to the original scale
            predictions = np.exp(predictions_log)

            # Calculate RMSE for the current fold
            rmse = np.sqrt(np.mean((test_fold["Close"].values - predictions) ** 2))
            rmse_scores.append(rmse)

            self.logger.info(f"Fold {fold_idx + 1}/{n_splits} RMSE: {rmse:.4f}")

        # Log the average RMSE across all folds
        self.logger.info(f"Cross-validation completed. Average RMSE: {np.mean(rmse_scores):.4f}")

        return rmse_scores

    def train(self):
        """
        Trains the ARIMA model on the training data.
        """
        train_data_log = np.log(self.train_data["Close"])

        # Test stationarity
        p_value = self.test_stationarity(train_data_log)
        if p_value > 0.05:
            self.differencing_order = 2
            self.logger.info("Time series is non-stationary. Differencing may be required.")
        else:
            self.differencing_order = None
            self.logger.info("Time series is stationary. Differencing may not be required.")

        # Train the ARIMA model
        self.model = auto_arima(
            train_data_log,
            exogenous=self.train_exog,
            start_p=0,
            start_q=0,
            max_p=self.max_p,
            max_q=self.max_q,
            d=self.differencing_order,
            test='adf',
            seasonal=False,
            suppress_warnings=True,
            error_action="ignore",
            trace=True,
            stepwise=True,
            n_fits=50,
            # index=train_data_log.index
        )

    def forecast(self):
        """
        Generates predictions for the entire test data using rolling forecast.

        Returns:
            pd.DataFrame: DataFrame containing actual and predicted values for the test data.
        """
        self.logger.info("Starting forecast...")
        if self.model is None:
            self.logger.error("Model has not been trained yet. Call `train()` first.")
            raise ValueError("Model has not been trained yet. Call `train()` first.")

        # Initialize predictions and history
        predictions = []

        # Predict for the entire test set
        if self.test_exog is not None:
            if len(self.test_exog) != len(self.test_data):
                self.logger.error("Exogenous variables and test data length mismatch.")
                raise ValueError("Exogenous variables must align with the length of the test data.")

            self.logger.info("Using exogenous variables for forecasting.")
            predictions_log = self.model.predict(n_periods=len(self.test_data), X=self.test_exog)
        else:
            self.logger.info("No exogenous variables provided. Forecasting univariate time series.")
            predictions_log = self.model.predict(n_periods=len(self.test_data))

        # Transform back to original scale (from log if applicable)
        predictions = np.exp(predictions_log.values)

        # Create a DataFrame of actual vs predicted values
        forecast_df = pd.DataFrame({
            "Date": self.test_data.index,
            "Predicted Close": predictions,
        })

        return forecast_df

    def save_model(self):
        """
        Saves the trained model as a pickle file.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_arima_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def save_predictions(self, forecast_df):
        """
        Saves the predictions as a CSV file.

        Parameters:
            forecast_df (pd.DataFrame): DataFrame containing predictions.
        """
        prediction_dir = "Output_Data/saved_predictions"
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, f"ARIMA_{self.stock_name}_predictions.csv")
        forecast_df.to_csv(prediction_path, index=False)

    def run(self):
        """
        Runs the full pipeline: tests stationarity, trains the model, generates forecasts,
        saves the model and predictions.
        """
        print(f"Training ARIMA model for {self.stock_name}...")
        self.train()
        self.validate_model()
        forecast_df = self.forecast()
        self.save_model()
        self.save_predictions(forecast_df)
        print(f"ARIMA model and predictions saved for {self.stock_name}.")
        return forecast_df
