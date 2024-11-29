import panel as pn
from static.config import MAIN_MAX_WIDTH, SCROLLABLE_STYLE, SIDEBAR_WIDTH

def create_layout(sidebar, tabs):
    """
    Creates the application layout with a sidebar and tabs.
    """
    template = pn.template.BootstrapTemplate(
        title="Stock Prices Dashboard",
        main_max_width=MAIN_MAX_WIDTH,
    )

    # Sidebar
    template.sidebar.append(sidebar.view())
    template.sidebar[-1].margin = 0

    # Main content with scrollable style
    template.main.append(
        pn.Tabs(*tabs, margin=0, sizing_mode="stretch_width")
    )
    template.main[0].style = SCROLLABLE_STYLE
    return template
