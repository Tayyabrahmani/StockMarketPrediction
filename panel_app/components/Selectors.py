import panel as pn
import param
import os
from pathlib import Path

class StockSelector(param.Parameterized):  
    stock_list = os.listdir(os.path.join(Path(__file__).parents[2], 'Input_Data', 'Processed_Files_Step1'))
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
        objects=["MLP", "RNN", "LSTM", "GRU", "CNN", "SVR", "ARIMA", "SARIMA", "Prophet"],
    )

    def view(self):
        return pn.Param(
            self.param.model_selector,
            widgets={
                "model_selector": pn.widgets.MultiSelect(
                    name="Select Models",
                    size=len(self.param.model_selector.objects)
                )
            },
            width=300,
            name="Model Selector",
        )
