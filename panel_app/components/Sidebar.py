import panel as pn
import param
from components.Selectors import ModelSelector, StockSelector
from static.config import SIDEBAR_WIDTH

class Sidebar(param.Parameterized):
    """
    Sidebar module for the dashboard, including stock and model selectors.
    """
    stock_selector = param.ClassSelector(class_=StockSelector)
    model_selector = param.ClassSelector(class_=ModelSelector)

    def __init__(self, stock_selector, model_selector, **params):
        super().__init__(**params)
        self.stock_selector = stock_selector
        self.model_selector = model_selector      

    def view(self):
        return pn.Column(
            pn.pane.HTML("<div class='sidebar'><h2>Controls</h2></div>"),
            pn.pane.Markdown("### Select Stock"),
            self.stock_selector.view(),
            self.model_selector.view(),
            width=SIDEBAR_WIDTH,
            margin=0,
            sizing_mode="stretch_height"
        )
