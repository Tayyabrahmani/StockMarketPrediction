import param
import panel as pn
import pandas as pd
from components.TimeSeriesPlot import TimeSeriesPlot
from components.ForecastTable import ForecastTable
from components.MachineLearningFramework import MachineLearningFramework
import os
from components.prediction_duration import PredictionDuration
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
        self.prediction_duration = PredictionDuration()
        self.prediction_duration_widget = self.prediction_duration.get_widget()

        # Initialize components
        self.time_series_plot = TimeSeriesPlot(
            stock_selector=self.stock_selector,
            model_selector=self.model_selector,
            prediction_duration=self.prediction_duration
        )

        self.forecast_table = ForecastTable(
            stock_selector=self.stock_selector,
            model_selector=self.model_selector,
            prediction_duration=self.prediction_duration
        )

        # Watch for changes in stock or model selection and update components
        self.stock_selector.param.watch(self._update_components, 'stock')
        self.model_selector.param.watch(self._update_components, 'model_selector')

        # Train and Predict button
        self.train_button = pn.widgets.Button(name="Train and Predict", button_type="primary")
        self.train_button.on_click(self.train_and_predict)

        # Initialize layout with components
        self._layout = pn.Column(
            self.time_series_plot.get_component(),
            self.forecast_table.get_component(),
            self.train_button,
            margin=20,
        )

    def train_and_predict(self, event=None):
        """
        Trains the selected machine learning models, stores predictions in Output_Data,
        and updates the time series plot and forecast table.
        """
        # Get selected stock and models
        stock_name = self.stock_selector.stock
        selected_models = self.model_selector.model_selector

        # Ensure stock and models are selected
        if not stock_name or not selected_models:
            pn.state.notifications.error("Please select a stock and at least one model.")
            return

        # Process data
        file_path = f"Input_Data/Processed_Files_Step2/{stock_name}.csv"
        if not os.path.exists(file_path):
            pn.state.notifications.error(f"File for stock {stock_name} not found.")
            return

        try:
            # Initialize and run the Machine Learning Framework
            self.ml_framework = MachineLearningFramework(file_path=file_path, stock_name=stock_name)
            self.ml_framework.select_models(selected_models)
            self.ml_framework.run_models()

            # Notify user of success
            if pn.state.notifications:
                pn.state.notifications.success(f"Training and predictions completed for {stock_name}.")
            else:
                print(f"Training and predictions completed for {stock_name}.")
        except Exception as e:
            pn.state.notifications.error(f"Error during training: {str(e)}")

        # Update components
        self._update_components()

    def _update_components(self, event=None):
        """
        Trigger updates for both the time series plot and forecast table.
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
