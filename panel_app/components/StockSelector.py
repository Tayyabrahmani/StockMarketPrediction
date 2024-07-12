import param
import panel as pn
import os
from pathlib import Path

class StockSelector(param.Parameterized):  
    stock_list = os.listdir(os.path.join(Path(__file__).parents[2], 'Input_Data', 'Processed_Files_Step1'))
    stock_list = [i.replace('.csv', '') for i in stock_list]
    stock = param.ObjectSelector(default=stock_list[0], objects=stock_list)

    def view(self):
        return pn.Param(self.param.stock, widgets={'stock': pn.widgets.Select})
