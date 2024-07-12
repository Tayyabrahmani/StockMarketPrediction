import panel as pn
import param
from components.TimeSeriesPlot import create_plot
from components.StockSelector import StockSelector
from pages.tabs import time_series_tab, errors_metrics_tab
from templates.layout import create_layout

class Dashboard(param.Parameterized):
    stock_selector = param.ClassSelector(class_=StockSelector)
    plot_pane = param.ClassSelector(class_=pn.pane.Plotly, default=pn.pane.Plotly())
    
    def __init__(self, **params):
        super().__init__(**params)
        self.stock_selector = StockSelector()
        self.update_plot()

    @param.depends('stock_selector.stock', watch=True)
    def update_plot(self):
        stock_name = self.stock_selector.stock
        new_plot = create_plot(stock_name)
        self.plot_pane.object = new_plot

    def view(self):
        return create_layout(self.stock_selector, self.plot_pane)

def create_dashboard():
    dashboard = Dashboard()
    return dashboard.view()
