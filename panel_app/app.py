import panel as pn
# Enable the Panel extension
pn.extension('plotly', 'tabulator', 'bootstrap')

from components.Selectors import ModelSelector, StockSelector
from components.Sidebar import Sidebar
from pages.PageFactory import create_pages
from templates.layout import create_layout


def run_app():
    # Create a shared StockSelector instance
    stock_selector = StockSelector()
    model_selector = ModelSelector()

    # Create the sidebar instance
    sidebar = Sidebar(stock_selector, model_selector)

    # Create the pages dynamically
    tabs = create_pages(stock_selector, model_selector, sidebar)

    # Create the complete layout
    template = create_layout(sidebar, tabs)

    # Serve the Panel application
    pn.serve(template, show=True, port=5000)

if __name__ == '__main__':
    run_app()