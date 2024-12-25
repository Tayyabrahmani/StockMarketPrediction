import plotly.graph_objects as go
from plotly.subplots import make_subplots
import panel as pn
import param
import pandas as pd
import os
from pathlib import Path    
from components.Selectors import StockSelector, ModelSelector

from config.model_config import AVAILABLE_MODELS

class TimeSeriesPlot(param.Parameterized):
    stock_selector = param.ClassSelector(class_=StockSelector, allow_refs=True)
    model_selector = param.ClassSelector(class_=ModelSelector, allow_refs=True)
    chart_type = param.Selector(objects=["Line Plot", "Candlestick"], default="Line Plot")
    duration = param.Selector(objects=["1M", "3M", "6M", "1Y", "5Y", "All"], default="All")
    plot_pane = param.ClassSelector(class_=pn.pane.Plotly, default=pn.pane.Plotly())
    prediction_duration = param.ClassSelector(class_=param.Parameterized, allow_None=True, allow_refs=True)

    def __init__(self, stock_selector, model_selector, prediction_duration, **params):
        super().__init__(**params)
        self.stock_selector = stock_selector
        self.model_selector = model_selector
        self.prediction_duration = prediction_duration
        self.prediction_duration_widget = prediction_duration.get_widget()
        self.plot_pane = pn.pane.Plotly(sizing_mode="stretch_width")

        # Initialize widgets
        self.chart_type_selector = pn.widgets.Select(
            name="Chart Type",
            options=["Line Plot", "Candlestick"],
            value=self.chart_type
        )
        self.duration_selector = pn.widgets.Select(
            name="Duration",
            options=["1M", "3M", "6M", "1Y", "5Y", "All"],
            value=self.duration
        )
        self.include_predictions_checkbox = pn.widgets.Checkbox(
            name="Include Predictions",
            value=True
        )

        # Link widget updates
        self.chart_type_selector.param.watch(self._update_chart_type, 'value')
        self.duration_selector.param.watch(self._update_duration, 'value')
        self.include_predictions_checkbox.param.watch(self.update_plot, 'value')
        self.model_selector.param.watch(self.update_plot, 'model_selector')

        # Ensure widgets are initialized before calling update_plot
        self.initialized = False
        self.update_plot()
        self.initialized = True

    def _update_chart_type(self, event):
        self.chart_type = event.new

    def _update_duration(self, event):
        self.duration = event.new

    def _update_prediction_duration(self, event):
        if self.initialized:
            self.update_plot()

    @param.depends('stock_selector.stock', 'chart_type', 'duration', 'prediction_duration.prediction_duration', watch=True)
    def update_plot(self, event=None):
        # Ensure the widget is initialized before accessing
        if not hasattr(self, 'include_predictions_checkbox'):
            return

        stock_name = self.stock_selector.stock
        if stock_name:
            stock_data = self.load_stock_data(stock_name)
            if stock_data is not None:
                filtered_data = self.filter_by_duration(stock_data)

                # Load predictions only if the checkbox is checked
                if self.include_predictions_checkbox.value:
                    predictions = self.load_predictions(stock_name)
                    filtered_predictions = self.filter_predictions(predictions)
                else:
                    filtered_predictions = pd.DataFrame()

                if self.chart_type == "Line Plot":
                    self.plot_pane.object = self.create_line_plot(filtered_data, filtered_predictions, stock_name)
                elif self.chart_type == "Candlestick":
                    self.plot_pane.object = self.create_candlestick_plot(filtered_data, stock_name)
            else:
                self.plot_pane.object = None
        else:
            self.plot_pane.object = None

    def load_stock_data(self, stock_name):
        try:
            file_path = os.path.join(
                Path(__file__).parents[2],
                "Input_Data",
                "Processed_Files_Step1",
                f"{stock_name}.csv"
            )
            stock_data = pd.read_csv(file_path)
            stock_data['Exchange Date'] = pd.to_datetime(stock_data['Exchange Date'])
            return stock_data
        except FileNotFoundError:
            return None

    def load_predictions(self, stock_name):
        if not self.stock_selector or not self.stock_selector.stock:
            return pd.DataFrame()

        # Get the selected models from the ModelSelector
        selected_models = self.model_selector.model_selector

        if not selected_models:
            return pd.DataFrame()

        predictions_dir = os.path.join(
            Path(__file__).parents[2],
            "Output_Data",
            "saved_predictions"
        )
        predictions = []

        for model in selected_models:
            file_path = os.path.join(predictions_dir, f"{model}_{stock_name}_predictions.csv")
            if os.path.exists(file_path):
                pred_data = pd.read_csv(file_path)
                pred_data['Model'] = model
                pred_data['Date'] = pd.to_datetime(pred_data['Date'])
                predictions.append(pred_data)

        combined_predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
        return combined_predictions

    def filter_by_duration(self, data, for_predictions=False):
        if data.empty or self.duration == "All":
            return data

        duration_mapping = {
            "1M": pd.Timedelta(days=30),
            "3M": pd.Timedelta(days=90),
            "6M": pd.Timedelta(days=180),
            "1Y": pd.Timedelta(days=365),
            "5Y": pd.Timedelta(days=1825),
        }
        cutoff_date = data['Date' if for_predictions else 'Exchange Date'].max() - duration_mapping[self.duration]
        return data[data['Date' if for_predictions else 'Exchange Date'] >= cutoff_date]

    def filter_predictions(self, predictions):
        """
        Filters predictions based on the prediction duration selector.
        """
        if predictions.empty or self.prediction_duration.prediction_duration == "All":
            return predictions

        duration_mapping = {
            "30 Days": pd.Timedelta(days=30),
            "60 Days": pd.Timedelta(days=60),
            "90 Days": pd.Timedelta(days=90),
        }

        # Ensure prediction_duration is valid
        prediction_duration = self.prediction_duration.prediction_duration
        if prediction_duration not in duration_mapping:
            return predictions

        cutoff_date = pd.Timestamp.now() + duration_mapping[prediction_duration]
        filtered_predictions = predictions[predictions["Date"] <= cutoff_date]
        return filtered_predictions

    def create_line_plot(self, stock_data, predictions, stock_name):
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )

        fig.add_trace(
            go.Scatter(
                x=stock_data["Exchange Date"],
                y=stock_data["Close"],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=1),
                marker=dict(size=2),
                name="Close Price",
                hovertemplate="<b>Date:</b> %{x}<br><b>Close:</b> %{y}"
            ),
            row=1, col=1
        )

        if not predictions.empty:
            for model_name, model_data in predictions.groupby("Model"):
                fig.add_trace(
                    go.Scatter(
                        x=model_data["Date"],
                        y=model_data["Predicted Close"],
                        mode="lines",
                        line=dict(dash="dash"),
                        name=f"{model_name} Prediction"
                    ),
                    row=1, col=1
                )

        fig.add_trace(
            go.Bar(
                x=stock_data["Exchange Date"],
                y=stock_data["Volume"],
                name="Volume",
                marker=dict(color="black"),
                hovertemplate="<b>Date:</b> %{x}<br><b>Volume:</b> %{y}"
            ),
            row=2, col=1
        )

        fig.update_layout(
            title={
                "text": f"<b>Stock Prices and Volume for {stock_name} ({self.duration})</b>",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(size=15, family="Arial, sans-serif", color="Black")
            },
            xaxis=dict(title="Date", showgrid=True, gridcolor="LightGray"),
            yaxis=dict(title="Close Price", showgrid=True, gridcolor="LightGray"),
            xaxis2=dict(title="Date"),
            yaxis2=dict(
                title="Volume",
                range=[0, stock_data["Volume"].max() * 1.1],
                showgrid=True,
                gridcolor="LightGray"
            ),
            template="plotly_white",
            height=450,
            margin=dict(l=20, r=20, t=80, b=20),
        )
        return fig

    def get_component(self):
        """
        Returns the complete layout for the time series plot.
        """
        return pn.Card(
            pn.Column(
                pn.Row(
                    self.chart_type_selector,
                    self.duration_selector,
                    self.include_predictions_checkbox,
                    self.prediction_duration_widget,  # Place widget in layout
                ),
                self.plot_pane,
                margin=10,
                sizing_mode="stretch_width",
            ),
            title="Time Series Plot",
            sizing_mode="stretch_width",
            height=600,
        )