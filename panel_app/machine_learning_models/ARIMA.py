import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
import os
import pickle
from machine_learning_models.preprocessing import load_data, preprocess_data_for_arima, train_test_split_time_series
from joblib import Parallel, delayed

class ARIMAStockModel:
    def __init__(self, file_path, stock_name, max_p=5, max_q=5, test_size=0.05):
        """
        Initializes the ARIMA model for stock prediction.

        Parameters:
            file_path (str): Path to the CSV file containing stock data.
            stock_name (str): Name of the stock for saving models and predictions.
            forecast_period (int): Number of future periods to forecast.
            max_p (int): Maximum AR parameter for auto_arima.
            max_q (int): Maximum MA parameter for auto_arima.
            test_size (float): Fraction of the data to use for testing.
        """
        self.file_path = file_path
        self.stock_name = stock_name
        self.max_p = max_p
        self.max_q = max_q
        self.test_size = test_size
        self.model = None

        # Load raw data
        self.data = load_data(self.file_path)

        # Extract target column (Close) for ARIMA modeling
        self.target = preprocess_data_for_arima(self.data, target_col='Close')

        # Convert index and target into numpy arrays for splitting
        indices = self.data.index.to_numpy().reshape(-1, 1)
        target_values = self.target.to_numpy()

        # Split into train/test sets using preprocessing.py
        train_indices, test_indices, train_values, test_values = train_test_split_time_series(
            indices, target_values, test_size=self.test_size
        )

        # Convert train/test splits back to DataFrame for ARIMA compatibility
        self.train_data = pd.DataFrame(train_values, index=pd.to_datetime(train_indices.flatten()), columns=['Close'])
        self.test_data = pd.DataFrame(test_values, index=pd.to_datetime(test_indices.flatten()), columns=['Close'])

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

    def train(self):
        """
        Trains the ARIMA model on the training data.
        """
        train_data_log = np.log(self.train_data[['Close']])

        # Test stationarity
        p_value = self.test_stationarity(train_data_log)
        if p_value > 0.05:
            differenced_data = train_data_log.diff().dropna()
            self.test_stationarity(differenced_data)
        else:
            differenced_data = train_data_log

        # # Ensure the training data has a DatetimeIndex
        # train_data_log.index = pd.to_datetime(train_data_log.index)

        # Train the ARIMA model
        self.model = auto_arima(
            differenced_data,
            start_p=0,
            start_q=0,
            max_p=self.max_p,
            max_q=self.max_q,
            d=None,
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
        Generates predictions for the entire test data using rolling forecasting.

        Returns:
            pd.DataFrame: DataFrame containing actual and predicted values for the test data.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call `train()` before forecasting.")

        # Initialize predictions
        predictions = []
        history = list(self.train_data['Close'])
        test_data_log = np.log(self.test_data['Close'])
        # test_points = list(self.test_data['Close'])

        # Parallelize rolling forecasts
        # predictions = Parallel(n_jobs=-1)(delayed(self.rolling_forecast)(self.model, history, test) for test in test_points)

        # # Rolling forecast for the entire test data
        # for t in range(len(test_data_log)):
        #     model = self.model.fit(y=np.log(history))
        #     forecast = model.predict(n_periods=1)
        #     forecast_value = np.exp(forecast[0])
        #     predictions.append(forecast_value)
        #     history.append(self.test_data['Close'].iloc[t])

        # Rolling forecast for the test data
        for actual_value in self.test_data['Close']:
            # Fit the model on the current history
            fitted_model = self.model.fit(y=np.log(history))

            # Forecast the next value
            forecast = fitted_model.predict(n_periods=1)
            forecast_value = np.exp(forecast[0])
            predictions.append(forecast_value)

            # Append the actual value to history for the next step
            history.append(actual_value)

        # Create a DataFrame of actual vs predicted values
        forecast_df = pd.DataFrame({
            "Date": self.test_data.index,
            "Predicted Close": predictions
        })
        return forecast_df

    # def rolling_forecast(self, model, history, test_point):
    #     fitted_model = model.fit(y=np.log(history))
    #     forecast = fitted_model.predict(n_periods=1)
    #     forecast_value = np.exp(forecast[0])
    #     return forecast_value

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
        self.test_stationarity(self.train_data["Close"])
        self.train()
        forecast_df = self.forecast()
        self.save_model()
        self.save_predictions(forecast_df)
        print(f"ARIMA model and predictions saved for {self.stock_name}.")
        return forecast_df

# if __name__ == "__main__":
#     # Initialize the ARIMA model
#     arima_model = ARIMAStockModel(
#         file_path='Input_Data/Processed_Files_Step1/Alphabet Inc.csv',
#         stock_name='Alphabet Inc',
#         forecast_period=60,
#         max_p=3,
#         max_q=3,
#         test_size=0.2
#     )

#     # Run the ARIMA pipeline
#     forecast = arima_model.run()
#     print(f"Forecast Results:\n{forecast}")