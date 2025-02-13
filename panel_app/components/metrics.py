import pandas as pd
import panel as pn
import plotly.express as px

class MetricsVisualizer:
    """
    A utility class for visualizing model evaluation metrics.
    """

    def __init__(self, metrics_file_path):
        """
        Initializes the MetricsVisualizer.

        Parameters:
            metrics_file (str): Path to the CSV file containing model metrics.
        """
        self.metrics_file_path = metrics_file_path
        self.data = self.load_metrics()

    def load_metrics(self):
        """
        Loads metrics from the CSV file.

        Returns:
            pd.DataFrame: The metrics data.
        """
        try:
            metrics_data = pd.read_csv(self.metrics_file_path)
            return metrics_data
        except FileNotFoundError:
            print(f"Metrics file not found: {self.metrics_file_path}")
            return pd.DataFrame()

    def filter_metrics(self, stock_name):
        """
        Filters metrics based on the selected stock name.

        Parameters:
            stock_name (str): The selected stock name.
        """
        if self.data.empty:
            return pd.DataFrame()
        return self.data[self.data["Stock"] == stock_name]

    def create_metrics_table(self, stock_name):
        """
        Creates a styled table displaying the metrics.

        Returns:
            pn.Card: A card containing the styled metrics table.
        """
        filtered_data = self.filter_metrics(stock_name)
        if filtered_data.empty:
            return pn.pane.Markdown("No metrics available for the selected stock.", style={"color": "red"})

        # Define column widths
        columns = {col: 150 for col in filtered_data.columns}

        # Create a Tabulator table with styling
        table = pn.widgets.Tabulator(
            filtered_data,
            layout="fit_data_stretch",
            selectable=False,
            theme="bootstrap4",
            height=300,
            widths=columns,
            show_index=False
        )

        # Wrap the table in a card for better appearance
        return pn.Card(
            table,
            title=f"Metrics Table for {stock_name}",
            collapsed=False,
            width=800,
        )

    def create_metrics_bar_chart(self, stock_name):
        """
        Creates a bar chart visualizing metrics by model.

        Returns:
            pn.Card: A card containing the bar chart.
        """
        filtered_data = self.filter_metrics(stock_name)
        if filtered_data.empty:
            return pn.Card(
                pn.pane.Markdown("No data available for the selected stock.", style={"color": "red"}),
                title=f"Bar Chart for {stock_name}",
                collapsed=False,
                sizing_mode="fixed",
                width=800,
            )

        # Dropdown widget for metric selection
        metric_selector = pn.widgets.Select(
            name="Select Metric",
            options=["RMSE", "MAE", "R²"],
            value="RMSE",
        )

        @pn.depends(metric_selector.param.value)
        def update_bar_chart(selected_metric):
            fig = px.bar(
                filtered_data,
                x="Model",
                y=selected_metric,
                title=f"{selected_metric} by Model for {stock_name}",
                labels={selected_metric: selected_metric, "Model": "Model"},
                color="Model",  # Use color to distinguish models
                template="plotly_white",
                text_auto=".2f",  # Display values at the top of the bars with 2 decimal precision
            )
            # Custom styling for the chart
            fig.update_layout(
                title={
                    "text": f"<b>{selected_metric} by Model for {stock_name}</b>",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                    "font": {"size": 20, "family": "Arial Black"},
                },
                xaxis=dict(
                    title="<b>Model</b>",
                    titlefont=dict(size=16),
                    tickfont=dict(size=14),
                    showline=True,
                    linewidth=2,
                    linecolor="black",
                ),
                yaxis=dict(
                    title=f"<b>{selected_metric}</b>",
                    titlefont=dict(size=16),
                    tickfont=dict(size=14),
                    showline=True,
                    linewidth=2,
                    linecolor="black",
                ),
                plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent background
                paper_bgcolor="rgba(240, 240, 255, 1)",  # Soft background color
                font=dict(family="Arial", size=12, color="black"),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial",
                ),
            )
            fig.update_traces(
                textfont_size=12,
                textposition="outside",  # Place text outside the bar
                marker=dict(
                    line=dict(width=1.5, color="black"),  # Add border to bars
                ),
            )
            return pn.pane.Plotly(fig, height=500)

        return pn.Card(
            pn.Column(metric_selector, update_bar_chart),
            title=f"Bar Chart for {stock_name}",
            collapsed=False,
            width=800,
        )

    def view(self, stock_name):
        """
        Combines the metrics table and bar chart into a single view for the selected stock.

        Parameters:
            stock_name (str): The selected stock name.

        Returns:
            pn.Column: A combined view of metrics table and bar chart.
        """
        return pn.Column(
            self.create_metrics_table(stock_name),
            self.create_metrics_bar_chart(stock_name),
            margin=(10, 10, 10, 10),
        )

    # def create_metrics_line_chart(self):
    #     """
    #     Creates a line chart for metric trends over time (if applicable).

    #     Returns:
    #         pn.pane.Plotly: A Plotly line chart.
    #     """
    #     if "Date" not in self.metrics_data.columns:
    #         return pn.pane.Markdown("**Date column not found in metrics data for line chart.**")
        
    #     fig = px.line(
    #         self.metrics_data,
    #         x="Date",
    #         y="MAE",
    #         color="Model",
    #         title="MAE Trends Over Time by Model",
    #     )
    #     return pn.pane.Plotly(fig)

