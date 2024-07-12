import panel as pn

def create_layout(stock_selector, plot_pane):
    sidebar = pn.Column(
        pn.pane.HTML("<div style='background-color:#f0f0f0; padding: 10px'><h2>Select Stock</h2></div>"),
        stock_selector.view(),
        width=300,
        margin=(10, 10, 10, 10),
        sizing_mode='stretch_height'
    )

    main_area = pn.Column(
        pn.pane.Markdown("# Stock Prices Dashboard", styles={'text-align': 'center'}),
        plot_pane,
        margin=(10, 10, 10, 10),
        sizing_mode='stretch_both'
    )
    

    template = pn.template.BootstrapTemplate(title='Stock Prices Dashboard')
    template.sidebar.append(sidebar)
    template.main.append(main_area)
    return template