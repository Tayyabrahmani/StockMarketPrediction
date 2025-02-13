from pages.TimeSeriesPage import create_time_series_page
from pages.MetricsPage import create_metrics_page

def create_pages(stock_selector, model_selector, sidebar):
    """
    Dynamically creates the application pages.
    """
    # Adjust sidebar visibility
    def create_time_series():
        return create_time_series_page(stock_selector, model_selector)

    def create_metrics():
        return create_metrics_page(stock_selector)

    return [
        ('Time Series', create_time_series()),
        ('Metrics', create_metrics()),
    ]
