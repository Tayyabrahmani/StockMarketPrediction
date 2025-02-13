import panel as pn
import param

class PredictionDuration(param.Parameterized):
    """
    A shared prediction duration selector to synchronize components.
    """
    prediction_duration = param.Selector(
        objects=["30 Days", "60 Days", "90 Days", "All"],
        default="30 Days",
    )

    def get_widget(self):
        widget = pn.widgets.Select(
            name="Prediction Duration",
            options=["30 Days", "60 Days", "90 Days", "All"],
            value=self.prediction_duration,
        )

        # Sync widget with `prediction_duration` parameter
        widget.link(self, value='prediction_duration')
        return widget
