import panel as pn
import param
from templates.layout import create_layout
from components.StockSelector import StockSelector

class MetricsPage(param.Parameterized):
    stock_selector = param.ClassSelector(class_=StockSelector, allow_refs=True)

    def __init__(self, stock_selector, **params):
        super().__init__(**params)
        self.stock_selector = stock_selector

    def view(self):
        return pn.Column(
            self.stock_selector.view(),
            self.create_metrics_errors_view(),
            margin=(10, 10, 10, 10),
        )

    def create_metrics_errors_view(self):
        return pn.Column(
            pn.pane.Markdown("# Metrics and Errors", styles={'text-align': 'center'}),
            pn.pane.Markdown("## Error Metrics will be displayed here"),
            margin=(10, 10, 10, 10),
        )

def create_metrics_page(stock_selector):
    metrics = MetricsPage(stock_selector)
    return metrics.view()