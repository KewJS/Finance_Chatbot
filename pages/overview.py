import sd_material_ui as sdmui
import dash
import dash_core_components as dcc
import dash_html_components as html
# from dash_html_components.Ol import Ol
import dash_bootstrap_components as dbc

from app import app

dash.register_page(__name__)

layout = dbc.Container([
    html.H1("RAK-Voice Overview"),
    html.Hr(),
    html.Br(),

    dbc.Card([
        html.Img(src=app.get_asset_url("./assets/rakvoice_overview.png"), className="center", style={"width": "100%"}),
        html.Br(),
        html.Img(src=app.get_asset_url("./assets/rakvoice_overview.png"), className="center", style={"width": "100%"}),
        html.Br(),
        html.Img(src=app.get_asset_url("./assets/rakvoice_solution.png"), className="center", style={"width": "100%"}),
    ], body=True)

], className="mt-4")