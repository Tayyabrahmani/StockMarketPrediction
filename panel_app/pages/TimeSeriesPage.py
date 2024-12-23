import param
import panel as pn
import pandas as pd
from components.TimeSeriesPlot import TimeSeriesPlot
from components.ForecastTable import ForecastTable
# from components.MachineLearningFramework import MachineLearningFramework
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

        # self.ml_framework = MachineLearningFramework()  # Initialize the ML framework

        # Train and Predict button
        # self.train_button = pn.widgets.Button(name="Train and Predict", button_type="primary")
        # self.train_button.on_click(self.train_and_predict)

        # Shared prediction duration widget
        # self.prediction_duration_widget = self.prediction_duration.get_widget()

        # Initialize layout with components
        self._layout = pn.Column(
            self.time_series_plot.get_component(),
            self.forecast_table.get_component(),
            # self.train_button,
            margin=20,
        )

    # def train_and_predict(self, event=None):
    #     """
    #     Trains the selected machine learning models, stores predictions in Output_Data,
    #     and updates the time series plot and forecast table.
    #     """
    #     # Get selected stock and models
    #     stock_name = self.stock_selector.stock
    #     selected_models = self.model_selector.model_selector

    #     # Ensure stock and models are selected
    #     if not stock_name or not selected_models:
    #         pn.state.notifications.error("Please select a stock and at least one model.")
    #         return

    #     # Process data
    #     file_path = f"Input_Data/Processed_Files_Step1/{stock_name}.csv"
    #     if not os.path.exists(file_path):
    #         pn.state.notifications.error(f"File for stock {stock_name} not found.")
    #         return

    #     data = pd.read_csv(file_path)
    #     data["Date"] = pd.to_datetime(data["Date"])
    #     data = data.set_index("Date")

    #     # Train and predict with each selected model
    #     for model_name in selected_models:
    #         try:
    #             # Set and train the model
    #             self.ml_framework.set_model(model_name)
    #             self.ml_framework.train(data, target_col="Close")

    #             # Save trained model as pickle
    #             self.ml_framework.save_model(stock_name=stock_name, model_name=model_name)

    #             # Generate and save predictions
    #             prediction_duration = self.time_series_plot.prediction_duration_selector.value
    #             prediction_periods = (
    #                 30 if prediction_duration == "30 Days" else
    #                 60 if prediction_duration == "60 Days" else
    #                 90 if prediction_duration == "90 Days" else
    #                 len(data)  # Default to all available data
    #             )
    #             predictions = self.ml_framework.predict(data, prediction_periods=prediction_periods)
    #             self.ml_framework.save_predictions(predictions, stock_name=stock_name, model_name=model_name)
    #         except Exception as e:
    #             pn.state.notifications.error(f"Error processing model {model_name}: {str(e)}")
    #             continue

    #     # Update components
    #     self.time_series_plot.update_plot()
    #     self.forecast_table.update_table()
    #     pn.state.notifications.success(f"Training and predictions completed for {stock_name}.")

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
