# forecast_table.py
import panel as pn
import param
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
pn.extension('tabulator')

class ForecastTable(param.Parameterized):
    """
    Displays a forecast table for a single stock and multiple models inside a card layout.
    """
    model_selector = param.ListSelector(default=["ARIMA"], objects=["ARIMA", "SARIMA", "Prophet"])
    stock_selector = param.ClassSelector(class_=param.Parameterized)
    result_message = param.String(default="Select a stock and one or more models to view forecasts.")
    base_dir = param.String(default="Output_Data/saved_predictions/")

    table = pn.widgets.Tabulator(
        pd.DataFrame({"Placeholder": ["No data available"]}),
        pagination="remote",
        page_size=15,
        sizing_mode="stretch_width",
        height=400
    )

    def __init__(self, **params):
        super().__init__(**params)
        self._message_pane = pn.pane.Markdown(self.result_message, margin=10)

    @param.depends("stock_selector.stock", "model_selector", watch=True)
    def update_table(self):
        stock_name = getattr(self.stock_selector, "stock", None)
        selected_models = self.model_selector

        if stock_name and selected_models:
            forecast_df = self.load_forecast(stock_name, selected_models[0])
            if not forecast_df.empty:
                self.table.value = forecast_df
                self.result_message = f"Forecast loaded for {stock_name}."
            else:
                self.table.value = pd.DataFrame({"Placeholder": ["No forecasts available"]})
                self.result_message = f"No forecasts available for {stock_name}."
        else:
            self.table.value = pd.DataFrame({"Placeholder": ["No data available"]})
            self.result_message = "No stock or models selected."

        self._message_pane.object = self.result_message

    def load_forecast(self, stock_name, model):
        file_path = f"{self.base_dir}/{model}_{stock_name}_forecast.csv"
        try:
            forecast_df = pd.read_csv(file_path)
            forecast_df = forecast_df.rename(
                columns={
                    "Forecast": f"{model} Forecast",
                    "Conf Int Lower": f"{model} Conf Int Lower",
                    "Conf Int Upper": f"{model} Conf Int Upper",
                }
            )
            return forecast_df
        except FileNotFoundError:
            logging.warning(f"Forecast file not found: {file_path}")
            return pd.DataFrame()

    def get_component(self):
        """
        Returns a Panel Card layout containing the forecast table and the result message.
        """
        return pn.Card(
            pn.Column(self.table, self._message_pane, margin=10),
            title="Forecast Table",
            sizing_mode="stretch_width",
            height=400,
        )
