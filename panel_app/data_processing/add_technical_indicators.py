import os
from pathlib import Path
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator, StochRSIIndicator, ROCIndicator, WilliamsRIndicator
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator, MACD, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import numpy as np

def add_technical_indicators(df):
    """
    Loads stock data from a CSV, calculates technical indicators, and saves the enhanced data to a new CSV.

    Parameters:
        input_csv (str): Path to the input CSV file containing stock data.
        output_csv (str): Path to the output CSV file to save the data with indicators.
    """
    df = df.ffill()

    # Add RSI
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi().shift(1)

    # Stochastic RSI
    df["Stoch"] = StochRSIIndicator(df["Close"], window=14, smooth1=3, smooth2=3).stochrsi().shift(1)
    
    # Add Stochastic Oscillator
    stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["Stochastic"] = stoch.stoch().shift(1)
    df['Stochastic_d'] = stoch.stoch_signal().shift(1)

    # Add MACD
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd().shift(1)
    df["MACD_Signal"] = macd.macd_signal().shift(1)
    df['macd_diff'] = macd.macd_diff().shift(1)

    # Add ROC
    df['roc'] = ROCIndicator(close=df['Close'], window=12).roc().shift(1)

    # Add SMA and EMA
    df["SMA_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator().shift(1)
    df["EMA_20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator().shift(1)

    # Add SMA and EMA
    df["SMA_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator().shift(1)
    df["EMA_50"] = EMAIndicator(close=df["Close"], window=50).ema_indicator().shift(1)

    # Add ADX
    df["ADX"] = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14).adx().shift(1)

    # Add Bollinger Bands
    bollinger = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["Bollinger_High"] = bollinger.bollinger_hband().shift(1)
    df["Bollinger_Low"] = bollinger.bollinger_lband().shift(1)
    df["Bollinger_Middle"] = bollinger.bollinger_mavg().shift(1)
    df['Bollinger_Width'] = bollinger.bollinger_wband().shift(1)

    # Add ATR
    df["ATR"] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range().shift(1)

    # Add OBV
    df["OBV"] = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume().shift(1)

    # Add OBV
    vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14)
    df['VWAP'] = vwap.volume_weighted_average_price().shift(1)

    # Calculate Pivot, Resistance, and Support using the previous 7 days' data
    df['Pivot'] = df[['High', 'Low', 'Close']].rolling(window=7).mean().mean(axis=1)
    df['Resistance'] = 2 * df['Pivot'] - df['Low'].rolling(window=7).mean().shift(1)
    df['Support'] = 2 * df['Pivot'] - df['High'].rolling(window=7).mean().shift(1)

    # Drop the 'Pivot' column as it's no longer needed
    df = df.drop(['Pivot'], axis=1)

    # Add CCI
    df['CCI'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci().shift(1)

    # Add williams_r
    df['Williams_r'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r().shift(1)

    # Add log of momentum
    # df['log_momentum'] = np.log(df["Close"] - 1)

    # Save the enhanced dataset
    return df

def process_all_stocks(input_dir, output_dir):
    """
    Processes all stock CSV files in the input directory, adds technical indicators,
    and saves the enhanced datasets to the output directory.

    Parameters:
        input_dir (str): Path to the directory containing input CSV files.
        output_dir (str): Path to the directory to save enhanced CSV files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each CSV file in the input directory
    for stock_file in input_path.glob("*.csv"):
        try:
            print(f"Processing file: {stock_file.name}")

            # Load stock data
            df = pd.read_csv(stock_file)
            
            # Ensure required columns are present
            required_columns = ["Exchange Date", "Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required_columns):
                print(f"Skipping {stock_file.name}: Missing required columns")
                continue
            
            # Sort data by date
            df = df.rename(columns={"Exchange Date": "Date"})
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")

            # Add technical indicators
            df = add_technical_indicators(df)

            df = df.rename(columns={"Date": "Exchange Date"})

            # Save enhanced data
            output_file = output_path / stock_file.name
            df.to_csv(output_file, index=False)
            print(f"Saved enhanced data to: {output_file}")
        
        except Exception as e:
            print(f"Error processing file {stock_file.name}: {e}")

# Example usage
if __name__ == "__main__":
    input_dir = "Input_Data/Processed_Files_Step1"
    output_dir = "Input_Data/Processed_Files_Step2"
    process_all_stocks(input_dir, output_dir)