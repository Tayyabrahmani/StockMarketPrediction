import panel as pn
import param

class ModelSelector(param.Parameterized):
    """
    A reusable model selector component.
    """
    model_selector = param.ListSelector(default=["ARIMA"], objects=["ARIMA", "SARIMA", "Prophet"])

    def view(self):
        return pn.Param(
            self.param.model_selector,
            widgets={"model_selector": pn.widgets.MultiSelect},
            width=250,
            name="Select Models",
        )
    