import panel as pn
import param
from components.metrics import MetricsVisualizer
from components.Selectors import StockSelector

class MetricsPage(param.Parameterized):
    stock_selector = param.ClassSelector(class_=StockSelector, allow_refs=True)
    metrics_file_path = param.String(default="Output_Data/model_metrics.csv")

    def __init__(self, stock_selector, metrics_file_path=None, **params):
        super().__init__(**params)
        self.stock_selector = stock_selector
        if metrics_file_path:
            self.metrics_file_path = metrics_file_path
        self.visualizer = MetricsVisualizer(metrics_file_path=self.metrics_file_path)

        # Watch for changes in the stock selector's value
        self.stock_selector.param.watch(self.update_visuals, 'stock')

        # Placeholder for dynamic metrics content
        self.metrics_content = pn.Column()

    def update_visuals(self, event=None):
        """
        Updates the metrics content based on the selected stock.
        """
        selected_stock = self.stock_selector.stock
        self.metrics_content.objects = [self.visualizer.view(selected_stock)]

    def view(self):
        """
        Renders the metrics page view.
        """
        selected_stock = self.stock_selector.stock
        self.metrics_content.objects = [self.visualizer.view(selected_stock)]
        return pn.Column(
            self.metrics_content,
            margin=(10, 10, 10, 10),
        )

def create_metrics_page(stock_selector):
    """
    Factory function to create a metrics page.
    """
    metrics_page = MetricsPage(stock_selector)
    return metrics_page.view()
