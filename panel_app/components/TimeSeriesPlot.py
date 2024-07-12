import plotly.express as px
import pandas as pd
import os
from pathlib import Path

def create_plot(stock_name):
    # Load the data
    file_path = os.path.join(Path(__file__).parents[2], "Input_Data", "Processed_Files_Step1", f'{stock_name}.csv')
    stock_data = pd.read_csv(file_path)

    # Convert the 'Exchange Date' to datetime format
    stock_data['Exchange Date'] = pd.to_datetime(stock_data['Exchange Date'])

    # Create the Plotly figure
    fig = px.line(
        stock_data,
        x='Exchange Date',
        y='Stock Price',
        title=f'Stock Prices of {stock_name}',
        labels={'Exchange Date': 'Date', 'Stock Price': 'Stock Price'},
        template='plotly'
    )

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Stock Price',
        showlegend=False
    )
    
    return fig
