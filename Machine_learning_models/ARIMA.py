import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from multiprocessing import Pool, cpu_count
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import pickle
import os

warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found. Using zeros as starting parameters.")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Exchange Date'] = pd.to_datetime(data['Exchange Date'], format='%Y-%m-%d')
    return data

def test_stationarity(timeseries):
    timeseries = timeseries.set_index('Exchange Date')
    timeseries = timeseries['Stock Price']
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # plt.figure(figsize=(10, 6))
    # plt.plot(timeseries, color='blue', label='Original')
    # plt.plot(rolmean, color='red', label='Rolling Mean')
    # plt.plot(rolstd, color='black', label='Rolling Std')
    # plt.legend(loc='best')
    # plt.title('Rolling Mean and Standard Deviation')
    # plt.show()

    print("Results of Dickey-Fuller Test:")
    adft = adfuller(timeseries.dropna(), autolag='AIC')
    output = pd.Series(adft[0:4], index=['Test Statistic', 'p-value', 'No. of Lags Used', 'Number of Observations Used'])
    for key, value in adft[4].items():
        output[f'Critical Value ({key})'] = value
    print(output)

def model_arima(data):
    data_arima = data.set_index('Exchange Date')
    data_arima = np.log(data_arima[['Stock Price']])

    model_autoARIMA = auto_arima(data_arima, start_p=0, start_q=0,
                                 test='adf', max_p=5, max_q=5, m=1, d=None,
                                 seasonal=False, start_P=0, D=0, trace=True,
                                 error_action='ignore', suppress_warnings=True, 
                                 stepwise=True, n_fits=10)

    forecast_period = 60
    log_forecast, log_conf_int = model_autoARIMA.predict(n_periods=forecast_period, return_conf_int=True)
    forecast_index = pd.date_range(start=data_arima.index[-1], periods=forecast_period + 1, closed='right')

    forecast = np.exp(log_forecast)
    conf_int = np.exp(log_conf_int)

    return forecast, conf_int, model_autoARIMA, forecast_index

def save_model_and_forecast(stock_name, model, forecast, conf_int, forecast_index):
    directory = "Output_Data"

    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/saved_models/{stock_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    forecast_df = pd.DataFrame({'Date': forecast_index,
                            'Forecast': forecast, 
                            'Conf Int Lower': conf_int[:, 0], 
                            'Conf Int Upper': conf_int[:, 1]})
    forecast_df.to_csv(f"{directory}/saved_predictions/ARIMA_{stock_name}_forecast.csv", index=False)

def process_stock_data(stock_name, data):
    stock_data = data[data['Stock Name'] == stock_name]
    print(f"\nProcessing stock data for: {stock_name}")
    try:
        test_stationarity(stock_data)
        forecast, conf_int, model, forecast_index = model_arima(stock_data)
        save_model_and_forecast(stock_name, model, forecast, conf_int, forecast_index)
        return stock_name, forecast, conf_int
    except Exception as e:
        print(f"Error processing {stock_name}: {e}")
        return stock_name, None, None

def main():
    data = load_data('Output_Data/cleaned_data.csv')
    stock_names = data['Stock Name'].unique()
    num_processes = min(len(stock_names), cpu_count())

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_stock_data, [(stock_name, data) for stock_name in stock_names])

    for stock_name, forecast, conf_int in results:
        if forecast is not None:
            print(f"\nForecast for {stock_name}:")
            print(forecast)
        else:
            print(f"\nNo forecast available for {stock_name}")

if __name__ == "__main__":
    main()
