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
        title={
            'text': f'Stock Prices of {stock_name}',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Date',
        yaxis_title='Stock Price',
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),  # Reduce margins to make it more compact
        autosize=True,  # Make the plot autosize
        height=500  # Set a fixed height for better responsiveness
    )

    fig.update_xaxes(
        rangeslider_visible=True,  # Add a range slider for better navigation
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1y', step='year', stepmode='backward'),
                dict(step='all')
            ])
        )
    )
 
    return fig
