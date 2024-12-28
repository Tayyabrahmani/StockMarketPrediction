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
    start_date_picker = pn.widgets.DatePicker(name="Start Date", value=None)
    end_date_picker = pn.widgets.DatePicker(name="End Date", value=None)
    reset_button = pn.widgets.Button(name="Reset Dates", button_type="primary")

    table = pn.widgets.Tabulator(
        pd.DataFrame({"Placeholder": ["No data available"]}),
        pagination="remote",
        page_size=15,
        sizing_mode="stretch_width",
        height=400,
        theme="materialize",
        layout="fit_data_table",
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

        # Watchers for date picker changes
        self.start_date_picker.param.watch(self.apply_date_filter, 'value')
        self.end_date_picker.param.watch(self.apply_date_filter, 'value')

        # Action for reset button
        self.reset_button.on_click(self.reset_dates)

        # Initialize the table with data
        self.update_table()

    def apply_date_filter(self, event=None):
        """
        Applies the date range filter to the Tabulator widget.
        """
        start_date = self.start_date_picker.value
        end_date = self.end_date_picker.value

        # Filter the DataFrame
        filtered_df = self.table.value

        if start_date:
            # Convert start_date to pandas.Timestamp for compatibility
            filtered_df = filtered_df[filtered_df["Date"] >= start_date]

        if end_date:
            # Convert end_date to pandas.Timestamp for compatibility
            filtered_df = filtered_df[filtered_df["Date"] <= end_date]

        # Update the Tabulator widget with the filtered DataFrame
        self.table.value = filtered_df

    def reset_dates(self, event=None):
        """
        Resets the date pickers and displays the full table.
        """
        self.start_date_picker.value = None
        self.end_date_picker.value = None
        self.update_table()  # Reset the table to show all rows

    @param.depends("stock_selector.stock", "model_selector.model_selector", watch=True)
    def update_table(self, event=None):
        """
        Updates the forecast table with actual values and forecasts for all selected models.
        """
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
        Loads and combines forecast data for the given stock and models, ensuring the first column 
        is the actual values, followed by forecast data for each model.
        """
        # Load actual values
        actual_values = self.load_actual_values(stock_name, prediction_days)
        if actual_values.empty:
            logging.warning(f"No actual values found for stock: {stock_name}")
            return pd.DataFrame()

        # Initialize the combined DataFrame with actual values
        combined_forecasts = actual_values

        for model in models:
            # Load forecast data for the current model
            forecast = self.load_forecast(stock_name, model, prediction_days)
            if not forecast.empty:
                # Rename "Predicted Close" to "{model} Forecast"
                forecast = forecast.rename(columns={"Predicted Close": f"{model} Forecast"})
                # Merge with the combined DataFrame on "Date"
                combined_forecasts = pd.merge(
                    combined_forecasts,
                    forecast,
                    on="Date",
                    how="left"  # Ensure all actual values are retained
                )

        # Round values and sort by date
        combined_forecasts = combined_forecasts.round(3)
        combined_forecasts = combined_forecasts.sort_values("Date", ascending=False)
        combined_forecasts["Date"] = combined_forecasts["Date"].dt.date

        return combined_forecasts

    def load_forecast(self, stock_name, model, prediction_days):
        """
        Loads forecast data for a specific stock and model, including actual values and predictions, 
        and filtering by prediction duration.
        """
        file_path = os.path.join(self.base_dir, f"{model}_{stock_name}_predictions.csv")
        try:
            forecast_df = pd.read_csv(file_path)
            if "Date" not in forecast_df or "Predicted Close" not in forecast_df:
                logging.warning(f"Invalid forecast file format: {file_path}")
                return pd.DataFrame()

            forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

            # Filter by prediction days
            if prediction_days:
                cutoff_date = pd.Timestamp.now() + pd.Timedelta(days=prediction_days)
                forecast_df = forecast_df[forecast_df["Date"] <= cutoff_date]

            return forecast_df[["Date", "Predicted Close"]]
        except FileNotFoundError:
            logging.warning(f"Forecast file not found: {file_path}")
            return pd.DataFrame()

    def load_actual_values(self, stock_name, prediction_days):
        """
        Loads actual close values for the given stock from the processed input data.
        """
        file_path = os.path.join("Input_Data/Processed_Files_Step1", f"{stock_name}.csv")
        try:
            actual_df = pd.read_csv(file_path)
            if "Exchange Date" not in actual_df or "Close" not in actual_df:
                logging.warning(f"Invalid processed file format: {file_path}")
                return pd.DataFrame()

            actual_df["Exchange Date"] = pd.to_datetime(actual_df["Exchange Date"])

            # Filter by prediction days
            if prediction_days:
                cutoff_date = pd.Timestamp.now() + pd.Timedelta(days=prediction_days)
                actual_df = actual_df[actual_df["Exchange Date"] <= cutoff_date]

            return actual_df[["Exchange Date", "Close"]].rename(columns={"Close": "Actual", "Exchange Date": "Date"})
        except FileNotFoundError:
            logging.warning(f"Processed file not found: {file_path}")
            return pd.DataFrame()
        
    def get_component(self):
        """
        Returns the component with added date filter functionality.
        """
        return pn.Card(
            pn.Column(
                self._message_pane,
                pn.Row(self.start_date_picker, self.end_date_picker,
                       pn.Column(pn.Spacer(height=30), self.reset_button),
                       margin=(0, 0, 10, 0)),
                self.table,
            ),
            title="Forecast Table",
            collapsible=True,
            sizing_mode="stretch_width",
            height=600,
        )