import panel as pn
from components.StockSelector import StockSelector
from pages.TimeSeriesPage import create_time_series_page
from pages.MetricsPage import create_metrics_page

# Load custom CSS from the static directory
css_file = "static/styles.css"
pn.config.css_files.append(css_file)

# Enable the Panel extension
pn.extension('plotly', 'tabulator', 'bootstrap')

def run_app():
    # Create a shared StockSelector instance
    stock_selector = StockSelector()

    # Create the time series and metrics views
    time_series_view = create_time_series_page(stock_selector)
    metrics_view = create_metrics_page(stock_selector)

    # Create tabs
    tabs = pn.Tabs(
        ('Time Series', time_series_view),
        ('Metrics', metrics_view),
        sizing_mode='stretch_both'
    )

    # Create sidebar and main content layout
    sidebar = pn.Column(
        pn.pane.HTML("<div class='sidebar'><h2>Select Stock</h2></div>"),
        stock_selector.view(),
        width=300,
        margin=(10, 10, 10, 10),
        sizing_mode='stretch_height'
    )

    template = pn.template.BootstrapTemplate(title='Stock Prices Dashboard')
    template.sidebar.append(sidebar)
    template.main.append(tabs)

    # Serve the Panel application
    pn.serve(template, show=True, port=5000)

if __name__ == '__main__':
    run_app()