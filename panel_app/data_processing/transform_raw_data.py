import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
import traceback

def get_stock_name(df):
    stock_name = df.iloc[0, 0].split("|")[0].strip()
    ticker_name = df.iloc[3, 0].split(".")[0].strip()
    return stock_name, ticker_name

def get_time_series_data(df, stock_name):
    # Get the index where we find Exchange Date value
    date_index = df[df.iloc[:, 0] == 'Exchange Date'].index

    # Remove all the rows before exchange date
    df = df.iloc[date_index[0]:].reset_index(drop=True)
    df = df.iloc[:, 0:10]

    # Make the first row the header
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    # Convert Exchange Date into date dtype
    df['Exchange Date'] = pd.to_datetime(df['Exchange Date'], format="%Y-%m-%d %H:%M:%S")

    # Order the time series to ascending
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Rename column
    df = df[['Exchange Date', 'Close', 'Open', 'Low', 'High', 'Volume']]
    # df = df.rename(columns={"Close": "Stock Price"})

    # Add stock name
    df['Stock Name'] = stock_name
    return df


def read_excel_files_from_directory(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter the list to include only Excel files
    excel_files = [file for file in files if file.endswith('.xls') or file.endswith('.xlsx')]
    
    # Initialize a list to store DataFrames
    dataframes = []
    
    # Read each Excel file into a pandas DataFrame
    for file in excel_files:
        try:
            file_path = os.path.join(directory, file)
            df = pd.read_excel(file_path, header=None)
            stock_name, ticker_name = get_stock_name(df)
            df_clean = get_time_series_data(df, stock_name)
            df_clean.to_csv(f"Input_Data/Processed_Files_Step1/{stock_name}.csv",index=False)
            dataframes.append(df_clean)

        except Exception as e:
            print(file)
            print(traceback.format_exc())

    return dataframes

if __name__ == '__main__':
    # Example usage
    directory = 'Input_Data/Raw_Files'  # Replace with your directory path
    all_dataframes = read_excel_files_from_directory(directory)
