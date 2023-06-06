import os
import base64
import pandas as pd
import dash_daq as daq
import time as time_pck
import sd_material_ui as sdmui
import plotly.graph_objects as go
from datetime import datetime, time, timedelta

import dash
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import Input, Output, State, html, dcc, dash_table, MATCH, ALL, ctx

from app import app, server

from pages.callbacks import *
import pages
from pages import *


app.title="RAK-Voice Project"
server = app.server


def create_main_nav_link(icon, label, href):
    return dcc.Link(
        dmc.Group(
            direction="row",
            position="center",
            spacing=10,
            style={"margin-bottom": 5},
            children=[
                dmc.ThemeIcon(
                    DashIconify(icon=icon, width=18),
                    size=25,
                    radius=5,
                    color="red",
                    variant="filled",
                    style={"margin-left":10}
                ),
                dmc.Text(label, size="sm", color="red", weight=700),
                ]
        ),
        href=href,
        style={"textDecoration": "none"},
    )
    
def create_accordianitem(icon, label, href):
    return dcc.Link(
        dmc.Group(
            direction="row",
            position="left",
            spacing=10,
            style={"margin-bottom":10},
            children=[
                dmc.ThemeIcon(
                    DashIconify(icon=icon, width=18),
                    size=30,
                    radius=30,
                    color="indigo",
                    variant="light",
                ),
                dmc.Text(label, size="sm", color="gray", style={"font-family":"IntegralCF-Regular"}),
            ]
        ),
        href=href,
        style={"textDecoration": "none"},
    )
    

navBar=dbc.Navbar(
    id="navBar",
    children=[],
    sticky="top",
    color="primary",
    className="navbar navbar-expand-lg navbar-dark bg-primary",
)

app.layout=dmc.MantineProvider(
    id="dark-moder", 
    withGlobalStyles=True, 
    children=[
        html.Div(
            children=[
                dmc.Header(
                    height=50,
                    fixed=True,
                    pl=0,
                    pr=0,
                    pt=0,
                    style={"background-color":"red", "color":"whitesmoke"},
                    children=[
                        dmc.Container(
                            fluid=True,
                            children=[
                                dmc.Group(
                                    position="apart",
                                    align="center",
                                    children=[
                                        dmc.Center(
                                            children=[
                                                dcc.Link(
                                                    dmc.ThemeIcon(
                                                        html.Img(src= "assets/rakvoice.png", style={"width":75}),
                                                        radius="sm",
                                                        size=44,
                                                        variant="filled",
                                                        color="blue",
                                                        ),
                                                    href=app.get_relative_path("/"),
                                                    ),
                                                ]
                                            ),
                                        dmc.Group(
                                            direction="row",
                                            position="right",
                                            align="center",
                                            spacing="md",
                                            children=[
                                                html.Div(id="indicatorbox", className="indicator-box",
                                                    children=[
                                                        html.Div(id="indicatorpulse", className="indicator-pulse", children=[]),
                                                        html.Span(id="liveindicator", className= "live-indicator", children=["LIVE"]),
                                                        ]
                                                    ),
                                                html.A(
                                                    dmc.ThemeIcon(
                                                        DashIconify(icon="mdi:linkedin")
                                                        ),
                                                    href="https://www.linkedin.com/in/jing-sheng-kew-a3a05b64/",
                                                    target ="_blank"
                                                    ),

                                                html.A(
                                                    dmc.ThemeIcon(
                                                        DashIconify(icon="mdi:github"),
                                                        color="dark"
                                                        ),
                                                    href="https://github.com/KewJS/Finance_Chatbot",
                                                    target ="_blank"
                                                    )
                                                ],
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                dmc.Modal(
                    id="the-modal",
                    overflow="inside",
                    size="xl",
                    children=[],
                    opened=False
                    ),

                dmc.Navbar(
                    fixed=True,
                    width={"base": 300},
                    pl="sm",
                    pr="xs",
                    pt=0,
                    hidden=True,
                    hiddenBreakpoint="sm",
                    children=[
                        dmc.ScrollArea(
                            offsetScrollbars=True,
                            type="scroll",
                            children=[
                                dmc.Divider(label="Contents", style={"marginBottom": 20, "marginTop": 5}),
                                dmc.Group(
                                    direction="column",
                                    children=[
                                        create_main_nav_link(
                                            icon="pajamas:issue-type-objective",
                                            label="Projects Overview",
                                            href=app.get_relative_path("/"),
                                            ),
                                        create_main_nav_link(
                                            icon="dashicons:format-chat",
                                            label="RAK-Voice Chatbot",
                                            href=app.get_relative_path("/chat"),
                                            ),
                                        create_main_nav_link(
                                            icon="icon-park-outline:ad-product",
                                            label="Products Recognition",
                                            href=app.get_relative_path("/ner"),
                                            ),
                                        create_main_nav_link(
                                            icon="ooui:text-summary-ltr",
                                            label="Sentiments Analysis",
                                            href=app.get_relative_path("/sentiments"),
                                            ),
                                        create_main_nav_link(
                                            icon="carbon:model-alt",
                                            label="Prediction",
                                            href=app.get_relative_path("/predictions"),
                                            ),
                                        ],
                                    ),
                                dmc.Divider(size="xl"),
                                dmc.Group(
                                    direction="column",
                                    align="center",
                                    position="center",\
                                    spacing="xs",
                                    children =[
                                        dmc.Text("Built By: Finance Disruptor", size="sm"),
                                        ]
                                    ),
                                ],
                            )
                        ],
                    ),
                dcc.Location(id="url"),
                dmc.MediaQuery(
                    largerThan="xs",
                    styles={"height":"100%", "margin-left":"300px", "margin-top":70},
                    children=[
                        html.Div(
                            id="content",
                            style={"margin-top":"70px"}
                            )
                        ],
                    ),
                ]
            )
        ]
    )


@app.callback(Output("content", "children"),
              [Input("url", "pathname")])
def display_content(pathname):
    page_name = app.strip_relative_path(pathname)
    if not page_name:
        return pages.home.layout
    elif page_name == "chat":
        return pages.chat.layout
    elif page_name == "overview":
        return pages.overview.layout
    elif page_name == "ner":
        return pages.ner.layout
    elif page_name == "sentiments":
        return pages.sentiments.layout
    elif page_name == "predictions":
        return pages.predictions.layout

    
@app.callback([Output(component_id="liveindicator", component_property="className"),
               Output(component_id="indicatorpulse", component_property="className")],
              [Input(component_id="interval", component_property="n_intervals")])
def update_indicator(n):
    return "live-indicator", "indicator-pulse"

# app.layout=html.Div([
#     dcc.Location(id="url", refresh=False),
#     html.Div([
#         navBar,
#         html.Div(id="pageContent")
#     ])
# ], id="table-wrapper")


# @app.callback(Output("pageContent", "children"),
#               [Input("url", "pathname")])
# def displayPage(pathname):
#     if pathname == "/":
#         return home.layout

#     if pathname == "/home":
#         return home.layout

#     if pathname == "/overview":
#         return overview.layout

#     else:
#         return error.layout

if __name__ == "__main__":
    app.run_server(debug=False)