import panel as pn
import param
import os
from pathlib import Path
import os
from config.model_config import AVAILABLE_MODELS

class StockSelector(param.Parameterized):  
    stock_list = os.listdir(os.path.join(Path(__file__).parents[2], 'Input_Data', 'Processed_Files_Step2'))
    stock_list = [i.replace('.csv', '') for i in stock_list]
    stock = param.ObjectSelector(default=stock_list[0], objects=stock_list)

    def view(self):
        return pn.Param(self.param.stock, widgets={'stock': pn.widgets.Select})

class ModelSelector(param.Parameterized):
    """
    A reusable model selector component for time-series prediction models.
    """
    model_selector = param.ListSelector(
        default=["ARIMA"],
        objects=AVAILABLE_MODELS,
    )

    def view(self):
        """
        Returns a Panel view with a MultiChoice widget for selecting models.
        """
        return pn.Param(
            self.param.model_selector,
            widgets={
                "model_selector": pn.widgets.MultiChoice(
                    name="Select Models",
                    options=self.param.model_selector.objects,
                    value=self.param.model_selector.default,
                    placeholder="Choose one or more models...",
                )
            },
            width=300,
            name="Model Selector",
        )
