import panel as pn
from pages.TimeSeriesPage import create_dashboard

# Enable the Panel extension
pn.extension('plotly', 'tabulator', 'bootstrap')

def run_app():
    # Create the dashboard layout
    dashboard = create_dashboard()

    # Serve the Panel application
    pn.serve(dashboard, show=True)

if __name__ == '__main__':
    run_app()
