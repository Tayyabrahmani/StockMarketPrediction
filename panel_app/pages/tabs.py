import panel as pn

def time_series_tab(dashboard):
    return pn.Column(
        pn.Row(pn.pane.Markdown('# Time Series Plot', styles={'background': '#f0f0f0', 'padding': '10px'})),
        dashboard.stock_selector.view(),
        dashboard.plot_pane
    )

def errors_metrics_tab():
    return pn.Column(
        pn.Row(pn.pane.Markdown('# Errors and Metrics', styles={'background': '#f0f0f0', 'padding': '10px'})),
        pn.pane.Markdown('## Error Metrics will be displayed here')
    )