import dash
import dash_bootstrap_components as dbc

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"

external_stylesheets = [dbc.themes.LITERA, dbc.themes.BOOTSTRAP, FONT_AWESOME]

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets)

server = app.server

app.config.suppress_callback_exceptions = True