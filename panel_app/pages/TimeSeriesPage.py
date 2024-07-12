import panel as pn
import param
from components.TimeSeriesPlot import create_plot
from components.StockSelector import StockSelector

class TimeSeriesPage(param.Parameterized):
    stock_selector = param.ClassSelector(class_=StockSelector)
    plot_pane = param.ClassSelector(class_=pn.pane.Plotly, default=pn.pane.Plotly())
    
    def __init__(self, stock_selector, **params):
        super().__init__(**params)
        self.stock_selector = stock_selector
        self.update_plot()

    @param.depends('stock_selector.stock', watch=True)
    def update_plot(self):
        stock_name = self.stock_selector.stock
        new_plot = create_plot(stock_name)
        self.plot_pane.object = new_plot

    def view(self):
        return pn.Column(
            self.stock_selector.view(),
            pn.Card(self.plot_pane, title="Stock Price Over Time", sizing_mode="stretch_both", css_classes=["card"]),
            margin=(10, 10, 10, 10),
            sizing_mode='stretch_both'
        )

def create_time_series_page(stock_selector):
    page = TimeSeriesPage(stock_selector)
    return page.view()
