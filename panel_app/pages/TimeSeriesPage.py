# Timeseriespage.py
import panel as pn
import pandas as pd
from components.TimeSeriesPlot import TimeSeriesPlot
from components.ForecastTable import ForecastTable
pn.extension('tabulator')

class TimeSeriesPage:
    """
    TimeSeriesPage integrates the TimeSeriesPlot and ForecastTable components
    and manages their layout and interactions.
    """

    def __init__(self, stock_selector, model_selector):
        # Initialize dependencies
        self.stock_selector = stock_selector
        self.model_selector = model_selector

        # Initialize components
        self.time_series_plot = TimeSeriesPlot(stock_selector=self.stock_selector)
        self.forecast_table = ForecastTable(
            stock_selector=self.stock_selector, model_selector=self.model_selector
        )

        # Ensure forecast table updates
        self.forecast_table.update_table()

        # Initialize layout with components
        self._layout = pn.Column(
            self.time_series_plot.get_component(),
            self.forecast_table.get_component(),
            margin=20,
        )

    def update(self):
        """
        Manually trigger updates for all components, if needed.
        """
        self.time_series_plot.update_plot()
        self.forecast_table.update_table()

    def view(self):
        """
        Returns the pre-initialized layout for the TimeSeriesPage.
        """
        return self._layout

def create_time_series_page(stock_selector, model_selector):
    page = TimeSeriesPage(stock_selector=stock_selector, model_selector=model_selector)
    return page.view()
