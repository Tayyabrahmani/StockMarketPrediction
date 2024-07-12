import panel as pn

def create_layout(tabs):
    sidebar = pn.Column(
        pn.pane.HTML("<div class='sidebar'><h2>Select Stock</h2></div>"),
        tabs[0].object.param.stock_selector.view(),
        width=300,
        sizing_mode='stretch_height'
    )

    # main_area = pn.Column(
    #     pn.pane.HTML("<div class='main-title'>Stock Prices Dashboard</div>"),
    #     pn.Card(plot_pane, title="Stock Price Over Time", sizing_mode="stretch_both", css_classes=["card"]),
    #     margin=(10, 10, 10, 10),
    #     sizing_mode='stretch_both'
    # )

    template = pn.template.BootstrapTemplate(title='Stock Prices Dashboard')
    template.sidebar.append(sidebar)
    template.main.append(pn.Tabs(*tabs, sizing_mode='stretch_both'))
    return template