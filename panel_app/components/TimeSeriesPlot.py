import plotly.express as px
import panel as pn
import param
import pandas as pd
import os
from pathlib import Path
from components.StockSelector import StockSelector

class TimeSeriesPlot(param.Parameterized):
    stock_selector = param.ClassSelector(class_=StockSelector)
    plot_pane = param.ClassSelector(class_=pn.pane.Plotly, default=pn.pane.Plotly())

    def __init__(self, stock_selector, **params):
        super().__init__(**params)
        self.stock_selector = stock_selector
        self.plot_pane = pn.pane.Plotly()
        self.update_plot()

    @param.depends('stock_selector.stock', watch=True)
    def update_plot(self):
        stock_name = self.stock_selector.stock
        if stock_name:
            self.plot_pane.object = self.create_plot(stock_name)
        else:
            self.plot_pane.object = None

    def create_plot(self, stock_name):
        try:
            file_path = os.path.join(
                Path(__file__).parents[2],
                "Input_Data",
                "Processed_Files_Step1",
                f"{stock_name}.csv"
            )
            stock_data = pd.read_csv(file_path)

            stock_data['Exchange Date'] = pd.to_datetime(stock_data['Exchange Date'])

            fig = px.line(
                stock_data,
                x="Exchange Date",
                y="Stock Price",
                title=f"Stock Prices of {stock_name}",
                labels={"Exchange Date": "Date", "Stock Price": "Stock Price"},
                template="plotly"
            )

            fig.update_layout(
                title={
                    "text": f"Stock Prices of {stock_name}",
                    "y": 0.9,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top"
                },
                xaxis_title="Date",
                yaxis_title="Stock Price",
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=20),
                height=400
            )

            return fig
        except FileNotFoundError:
            return None

    def get_component(self):
        """
        Returns the layout for integration into a larger page.
        """
        return pn.Card(
            pn.Column(
                self.plot_pane,
                margin=10,
                sizing_mode="stretch_width",
            ),
            title="Time Series Plot",
            sizing_mode="stretch_width",
            height=500,
        )
