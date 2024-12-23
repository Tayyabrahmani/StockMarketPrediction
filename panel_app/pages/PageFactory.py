from pages.TimeSeriesPage import create_time_series_page
from pages.MetricsPage import create_metrics_page

def create_pages(stock_selector, model_selector):
    """
    Dynamically creates the application pages.
    """
    return [
        ('Time Series', create_time_series_page(stock_selector, model_selector)),
        ('Metrics', create_metrics_page(stock_selector)),
    ]