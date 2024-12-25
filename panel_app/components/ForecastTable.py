# forecast_table.py
import os
import panel as pn
import param
import pandas as pd
import logging
from components.Selectors import StockSelector, ModelSelector

logging.basicConfig(level=logging.INFO)
pn.extension('tabulator')

class ForecastTable(param.Parameterized):
    """
    Displays a forecast table for a single stock and multiple models inside a card layout.
    """
    prediction_duration = param.ClassSelector(class_=param.Parameterized, allow_None=True, allow_refs=True)
    model_selector = param.ClassSelector(class_=ModelSelector)
    stock_selector = param.ClassSelector(class_=StockSelector)
    result_message = param.String(default="Select a stock and one or more models to view forecasts.")
    base_dir = param.String(default="Output_Data/saved_predictions/")

    table = pn.widgets.Tabulator(
        pd.DataFrame({"Placeholder": ["No data available"]}),
        pagination="remote",
        page_size=15,
        sizing_mode="stretch_width",
        height=400,
        theme="materialize",
        layout="fit_data_table",
        header_filters=True,
        show_index=False,
    )
    
    def __init__(self, stock_selector, model_selector, prediction_duration, **params):
        super().__init__(**params)
        self._message_pane = pn.pane.Markdown(self.result_message, margin=10)
        self.stock_selector = stock_selector
        self.model_selector = model_selector
        self.prediction_duration = prediction_duration

        # Add watchers to dynamically update the table on stock or model changes
        self.stock_selector.param.watch(self.update_table, 'stock')
        self.model_selector.param.watch(self.update_table, 'model_selector')

        # Initialize the table with data
        self.update_table()

    @param.depends("stock_selector.stock", "model_selector.model_selector", watch=True)
    def update_table(self, event=None):
        if self.prediction_duration is None:
            self.result_message = "Prediction duration is not set."
            self._message_pane.object = self.result_message
            return

        stock_name = self.stock_selector.stock
        selected_models = self.model_selector.model_selector

        if not stock_name or not selected_models:
            self.table.value = pd.DataFrame({"Placeholder": ["No data available"]})
            self.result_message = "No stock or models selected."
            self._message_pane.object = self.result_message
            return

        prediction_duration = self.prediction_duration.prediction_duration
        prediction_days = (
            30 if prediction_duration == "30 Days" else
            60 if prediction_duration == "60 Days" else
            90 if prediction_duration == "90 Days" else
            None  # All data
        )

        if stock_name and selected_models:
            combined_forecasts = self.load_combined_forecasts(stock_name, selected_models, prediction_days)
            if not combined_forecasts.empty:
                self.table.value = combined_forecasts
                self.result_message = f"Forecasts loaded for {stock_name} with models: {', '.join(selected_models)}."
            else:
                self.table.value = pd.DataFrame({"Placeholder": ["No forecasts available"]})
                self.result_message = f"No forecasts available for {stock_name}."
        else:
            self.table.value = pd.DataFrame({"Placeholder": ["No data available"]})
            self.result_message = "No stock or models selected."

        self._message_pane.object = self.result_message

    def load_combined_forecasts(self, stock_name, models, prediction_days):
        """
        Loads and combines forecast data for the given stock and models, limited to the prediction duration.
        """
        combined_forecasts = []
        for model in models:
            forecast = self.load_forecast(stock_name, model, prediction_days)
            if not forecast.empty:
                forecast = forecast.rename(columns={col: f"{col}_{model}" if col != "Date" else "Date" for col in forecast.columns})
                combined_forecasts.append(forecast)

        # Combine all model forecasts into a single DataFrame
        if combined_forecasts:
            combined_df = pd.concat(combined_forecasts, axis=1)
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()].copy()
            return combined_df
        return pd.DataFrame()

    def load_forecast(self, stock_name, model, prediction_days):
        """
        Loads forecast data for a specific stock and model, filtering by prediction duration.
        """
        file_path = os.path.join(self.base_dir, f"{model}_{stock_name}_predictions.csv")
        try:
            forecast_df = pd.read_csv(file_path)
            if "Date" not in forecast_df or "Predicted Close" not in forecast_df:
                logging.warning(f"Invalid forecast file format: {file_path}")
                return pd.DataFrame()

            # Rename columns dynamically
            forecast_df = forecast_df.rename(
                columns={
                    "Predicted Close": f"{model} Forecast",
                },
            )

            forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

            # Filter by prediction days
            if prediction_days:
                cutoff_date = pd.Timestamp.now() + pd.Timedelta(days=prediction_days)
                forecast_df = forecast_df[forecast_df["Date"] <= cutoff_date]

            return forecast_df[["Date", f"{model} Forecast"]]
        except FileNotFoundError:
            logging.warning(f"Forecast file not found: {file_path}")
            return pd.DataFrame()

    def get_component(self):
        return pn.Card(
            pn.Column(
                self._message_pane,
                self.table,
            ),
            title="Forecast Table",
            collapsible=True,
            sizing_mode="stretch_width",
            height=600,
        )
