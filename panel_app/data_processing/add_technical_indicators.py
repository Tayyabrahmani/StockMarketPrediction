import os
from pathlib import Path
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def add_technical_indicators(df):
    """
    Loads stock data from a CSV, calculates technical indicators, and saves the enhanced data to a new CSV.

    Parameters:
        input_csv (str): Path to the input CSV file containing stock data.
        output_csv (str): Path to the output CSV file to save the data with indicators.
    """
    # Add RSI
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()

    # Add Stochastic Oscillator
    stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["Stochastic"] = stoch.stoch()

    # Add MACD
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    # Add SMA and EMA
    df["SMA_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["EMA_20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()

    # Add ADX
    df["ADX"] = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14).adx()

    # Add Bollinger Bands
    bollinger = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["Bollinger_High"] = bollinger.bollinger_hband()
    df["Bollinger_Low"] = bollinger.bollinger_lband()

    # Add ATR
    df["ATR"] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()

    # Add OBV
    df["OBV"] = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()

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