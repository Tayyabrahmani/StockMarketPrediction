import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
import os
import pickle
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress warnings
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found. Using zeros as starting parameters.")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class ARIMAStockModel:
    def __init__(self, file_path, stock_name, forecast_period=60, max_p=3, max_q=3, test_size=0.2):
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
        self.forecast_period = forecast_period
        self.max_p = max_p
        self.max_q = max_q
        self.test_size = test_size
        
        self.data = self.load_data()
        self.train_data, self.test_data = self.split_data()
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

    def split_data(self):
        """
        Splits the data into train and test sets, ensuring the latest data is used for testing.

        Returns:
            tuple: (train_data, test_data)
        """
        split_index = int(len(self.data) * (1 - self.test_size))
        train_data = self.data.iloc[:split_index]
        test_data = self.data.iloc[split_index:]
        return train_data, test_data

    def test_stationarity(self, timeseries):
        """
        Tests the stationarity of the given time series using the Augmented Dickey-Fuller test.

        Parameters:
            timeseries (pd.Series): The time series to test.
        """
        adft = adfuller(timeseries.dropna(), autolag='AIC')
        print(f"Dickey-Fuller Test Results:\n"
              f"Test Statistic = {adft[0]:.4f}\n"
              f"p-value = {adft[1]:.4f}\n")

    def train(self):
        """
        Trains the ARIMA model on the training data.

        Returns:
            auto_arima: The trained ARIMA model.
        """
        train_data_log = np.log(self.train_data[['Close']])
        self.model = auto_arima(
            train_data_log, 
            start_p=0, start_q=0,
            max_p=self.max_p, max_q=self.max_q,
            d=None, test='adf',
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
            n_fits=50
        )
        return self.model

    def forecast(self):
        """
        Generates predictions using the trained ARIMA model.

        Returns:
            pd.DataFrame: DataFrame containing forecasted values and dates.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call `train()` before forecasting.")
        
        forecast_log, conf_int_log = self.model.predict(
            n_periods=self.forecast_period, return_conf_int=True
        )
        forecast = np.exp(forecast_log)
        conf_int = np.exp(conf_int_log)
        
        # Adjusted: Generate forecast dates
        last_date = self.train_data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.forecast_period
        )
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted Close': forecast
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
        
        prediction_path = os.path.join(prediction_dir, f"ARIMA_{self.stock_name}_forecast.csv")
        forecast_df.to_csv(prediction_path, index=False)

    def run(self):
        """
        Runs the full pipeline: tests stationarity, trains the model, generates forecasts,
        saves the model and predictions.
        """
        print(f"Processing ARIMA model for {self.stock_name}...")
        self.test_stationarity(self.train_data['Close'])
        self.train()
        forecast_df = self.forecast()
        self.save_model()
        self.save_predictions(forecast_df)
        print(f"ARIMA model and predictions saved for {self.stock_name}.")
        return forecast_df

if __name__ == "__main__":
    # Initialize the ARIMA model
    arima_model = ARIMAStockModel(
        file_path='Input_Data/Processed_Files_Step1/Alphabet Inc.csv',
        stock_name='Alphabet Inc',
        forecast_period=60,
        max_p=3,
        max_q=3,
        test_size=0.2
    )

    # Run the ARIMA pipeline
    forecast = arima_model.run()
    print(f"Forecast Results:\n{forecast}")