import os
import pandas as pd
import time as time_pck
import sd_material_ui as sdmui
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time, timedelta

import spacy
from spacy import displacy

import dash
import dash_daq as daq
# import dash_design_kit as ddk
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc, dash_table, MATCH, ALL, ctx

from app import app

def create_table(df):
    columns, values = df.columns, df.values
    header = [html.Tr([html.Th(col) for col in columns])]
    rows = [html.Tr([html.Td(cell) for cell in row]) for row in values]
    table = [html.Thead(header), html.Tbody(rows)]
    
    return table

colors = {"SEGMENT": "#85C1E9", "INVESTMENT": "#ff6961", "AUTO": "lightgreen", "CC SERVICE": "#ffe4b5", "FD": "lightpurple", "MORTGAGE": "lightpink", "NEW CC": "#b0c4de", "SERVICE": "#cd5c5c"}
options = {"ents": ["SEGMENT", "INVESTMENT", "AUTO", "CC SERVICE", "FD", "MORTGAGE", "NEW CC", "SERVICE"], "colors": colors}

ner_desc_df = pd.read_csv("data/preprocess/FinancialPhraseBank-v1.0/ner_descriptions.csv")

header = [
    html.Thead(
        html.Tr(
            [
                html.Th("Entities"),
                html.Th("Descriptions"),
            ]
        )
    )
]
row1 = html.Tr([html.Td(ner_desc_df["Entities"][0]), html.Td(ner_desc_df["Descriptions"][0])])
row2 = html.Tr([html.Td(ner_desc_df["Entities"][1]), html.Td(ner_desc_df["Descriptions"][1])])
row3 = html.Tr([html.Td(ner_desc_df["Entities"][2]), html.Td(ner_desc_df["Descriptions"][2])])
row4 = html.Tr([html.Td(ner_desc_df["Entities"][3]), html.Td(ner_desc_df["Descriptions"][3])])
row5 = html.Tr([html.Td(ner_desc_df["Entities"][4]), html.Td(ner_desc_df["Descriptions"][4])])
row6 = html.Tr([html.Td(ner_desc_df["Entities"][5]), html.Td(ner_desc_df["Descriptions"][5])])
row7 = html.Tr([html.Td(ner_desc_df["Entities"][6]), html.Td(ner_desc_df["Descriptions"][6])])
row8 = html.Tr([html.Td(ner_desc_df["Entities"][7]), html.Td(ner_desc_df["Descriptions"][7])])

body = [html.Tbody([row1, row2, row3, row4, row5, row6, row7, row8])]

nlp_rak = spacy.load("./data/models/ner-model-best")
text1 = "How do I become an Elite customer? What Investment products do you offer? Suggest to me a cc with air miles benefit. How do I apply for Auto Loan with rakbank?"
doc1 = nlp_rak(text1)
html_doc1 = displacy.render(doc1, style="ent", options=options)

text2 = "Can I open a trading account with Rakbank? What is the lowest rate you can offer for mortgage loan?"
doc2 = nlp_rak(text2)
html_doc2 = displacy.render(doc2, style="ent", options=options)

text3 = "I want a credit card with supermarket cashback. How do I get car loan from Rakbank. Also, I forgot my Raktoken how do I reset it?"
doc3 = nlp_rak(text3)
html_doc3 = displacy.render(doc3, style="ent", options=options)

text4 = "How do I become an Elite customer? What Investment products do you offer? How do I apply for Car Loan with rakbank?"
doc4 = nlp_rak(text4)
html_doc4 = displacy.render(doc4, style="ent", options=options)


layout = html.Div(
    style={"margin_top":"70px"},
    children=[
        dmc.Group(
            direction="column",
            grow=True,
            children=[
                dmc.Title(children="Products Entities Summarization", order=3, style={'text-align':'center', 'color' :'slategray'}),
                dmc.Divider(label="Overview", labelPosition="center", size="xl"),
                dmc.Group(
                    direction="row",
                    grow=True,
                    children=[
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"175px"},
                            children=[
                                dmc.Center(
                                    dmc.ThemeIcon(
                                        size=50,
                                        radius="xl",
                                        color="violet",
                                        variant="light",
                                        children=[DashIconify(icon="fluent:people-community-20-filled", width=30)]
                                    )
                                ),
                                dmc.Group(
                                    direction='column',
                                    position='center',
                                    spacing='xs',
                                    style={'margin-top':10},
                                    children=[
                                        dmc.Text('Total Queries', size='xs', color='dimmed'),
                                        dmc.Text(id="totalsentences", size="xl", weight=700),
                                        
                                    ]
                                )
                            ],
                        ),
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"175px"},
                            children=[
                                dmc.Center(
                                    dmc.ThemeIcon(
                                        size=50,
                                        radius="xl",
                                        color="yellow",
                                        variant="light",
                                        children=[DashIconify(icon="flat-color-icons:automotive", width=30)]
                                    )
                                ),
                                dmc.Group(
                                    direction='column',
                                    position='center',
                                    spacing='xs',
                                    style={'margin-top':10},
                                    children=[
                                        dmc.Text('Auto', size='xs', color='dimmed'),
                                        dmc.Text(id="auto_sum", size="xl", weight=700),
                                    ]
                                )
                            ],
                        ),
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"175px"},
                            children=[
                                dmc.Center(
                                    dmc.ThemeIcon(
                                        size=50,
                                        radius="xl",
                                        color="green",
                                        variant="light",
                                        children=[DashIconify(icon="material-symbols:credit-card", width=30)]
                                    )
                                ),
                                dmc.Group(
                                    direction='column',
                                    position='center',
                                    spacing='xs',
                                    style={'margin-top':10},
                                    children=[
                                        dmc.Text('CC Service', size='xs', color='dimmed'),
                                        dmc.Text(id="cc_service_sum", size="xl", weight=700),
                                        
                                    ]
                                )
                            ],
                        ),
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"175px"},
                            children=[
                                dmc.Center(
                                    dmc.ThemeIcon(
                                        size=50,
                                        radius="xl",
                                        color="green",
                                        variant="light",
                                        children=[DashIconify(icon="material-symbols:account-balance", width=30)]
                                    )
                                ),
                                dmc.Group(
                                    direction='column',
                                    position='center',
                                    spacing='xs',
                                    style={'margin-top':10},
                                    children=[
                                        dmc.Text('FD', size='xs', color='dimmed'),
                                        dmc.Text(id="fd_sum", size="xl", weight=700),
                                    ]
                                )
                            ],
                        ),
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"175px"},
                            children=[
                                dmc.Center(
                                    dmc.ThemeIcon(
                                        size=50,
                                        radius="xl",
                                        color="green",
                                        variant="light",
                                        children=[DashIconify(icon="fxemoji:stockchart", width=30)]
                                    )
                                ),
                                dmc.Group(
                                    direction='column',
                                    position='center',
                                    spacing='xs',
                                    style={'margin-top':10},
                                    children=[
                                        dmc.Text('Investment', size='xs', color='dimmed'),
                                        dmc.Text(id="investment_sum", size="xl", weight=700),
                                    ]
                                )
                            ],
                        ),
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"175px"},
                            children=[
                                dmc.Center(
                                    dmc.ThemeIcon(
                                        size=50,
                                        radius="xl",
                                        color="green",
                                        variant="light",
                                        children=[DashIconify(icon="material-symbols:house", width=30)]
                                    )
                                ),
                                dmc.Group(
                                    direction='column',
                                    position='center',
                                    spacing='xs',
                                    style={'margin-top':10},
                                    children=[
                                        dmc.Text('Mortgage', size='xs', color='dimmed'),
                                        dmc.Text(id="mortgage_sum", size="xl", weight=700),
                                    ]
                                )
                            ],
                        ),
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"175px"},
                            children=[
                                dmc.Center(
                                    dmc.ThemeIcon(
                                        size=50,
                                        radius="xl",
                                        color="green",
                                        variant="light",
                                        children=[DashIconify(icon="mdi:account-credit-card", width=30)]
                                    )
                                ),
                                dmc.Group(
                                    direction='column',
                                    position='center',
                                    spacing='xs',
                                    style={'margin-top':10},
                                    children=[
                                        dmc.Text('New CC', size='xs', color='dimmed'),
                                        dmc.Text(id="new_cc_sum", size="xl", weight=700),
                                    ]
                                )
                            ],
                        ),
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"175px"},
                            children=[
                                dmc.Center(
                                    dmc.ThemeIcon(
                                        size=50,
                                        radius="xl",
                                        color="green",
                                        variant="light",
                                        children=[DashIconify(icon="material-symbols:segment", width=30)]
                                    )
                                ),
                                dmc.Group(
                                    direction='column',
                                    position='center',
                                    spacing='xs',
                                    style={'margin-top':10},
                                    children=[
                                        dmc.Text('Segment', size='xs', color='dimmed'),
                                        dmc.Text(id="segment_sum", size="xl", weight=700),
                                    ]
                                )
                            ],
                        ),
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"175px"},
                            children=[
                                dmc.Center(
                                    dmc.ThemeIcon(
                                        size=50,
                                        radius="xl",
                                        color="green",
                                        variant="light",
                                        children=[DashIconify(icon="material-symbols:home-repair-service", width=30)]
                                    )
                                ),
                                dmc.Group(
                                    direction='column',
                                    position='center',
                                    spacing='xs',
                                    style={'margin-top':10},
                                    children=[
                                        dmc.Text('Service', size='xs', color='dimmed'),
                                        dmc.Text(id="service_sum", size="xl", weight=700),
                                    ]
                                )
                            ],
                        ),
                    ]
                ),
                dmc.Divider(label="Products Analysis", labelPosition="center", size="xl"),
                dmc.Group(
                    direction="row",
                    grow=True,
                    children=[
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"600px"},
                            children=[
                                dmc.Title("Products Summarization Across Time", order=4, style={'text-align':'center', 'color':'grey'}),
                                dcc.Dropdown(
                                    id="entities-checklist",
                                    options=[
                                        {"label": "AUTO", "value": "AUTO"},
                                        {"label": "CC SERVICE", "value": "CC SERVICE"},
                                        {"label": "FD", "value": "FD"},
                                        {"label": "INVESTMENT", "value": "INVESTMENT"},
                                        {"label": "MORTGAGE", "value": "MORTGAGE"},
                                        {"label": "NEW CC", "value": "NEW CC"},
                                        {"label": "SEGMENT", "value": "SEGMENT"},
                                        {"label": "SERVICE", "value": "SERVICE"},
                                    ],
                                    value=["AUTO", "CC SERVICE", "FD", "INVESTMENT", "MORTGAGE", "NEW CC", "SEGMENT", "SERVICE"],
                                    multi=True,
                                    ),
                                dcc.Graph(id="entities-periods"),
                                ]
                            ),
                        ]
                    ),
                dmc.Divider(label="Customer Sentence Analysis", labelPosition="center", size="xl"),
                # html.Div(dcc.Markdown([html_doc], dangerously_allow_html=True)),
                dmc.Group(
                    direction="row",
                    grow=True,
                    children=[
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"600px"},
                            children=[
                                dmc.Title("Named Entity Recognition Model", order=4, style={'text-align':'center', 'color':'grey'}),
                                dcc.Markdown(["1. {}".format(html_doc1)], dangerously_allow_html=True),
                                html.Br(),
                                dcc.Markdown(["2. {}".format(html_doc2)], dangerously_allow_html=True),
                                html.Br(),
                                dcc.Markdown(["3. {}".format(html_doc3)], dangerously_allow_html=True),
                                html.Br(),
                                dcc.Markdown(["4. {}".format(html_doc4)], dangerously_allow_html=True),
                                ]
                            ),
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"600px"},
                            children=[
                                dmc.Title("Product Entities Descriptions", order=4, style={'text-align':'center', 'color':'grey'}),
                                dmc.Table(header + body),
                                ]
                            ),
                        ]
                    ),
                dmc.Space(h=100)
            ]
        )
    ],
)

@app.callback(Output("totalsentences", "children"),
              Output("auto_sum", "children"),
              Output("cc_service_sum", "children"),
              Output("fd_sum", "children"),
              Output("investment_sum", "children"),
              Output("mortgage_sum", "children"),
              Output("new_cc_sum", "children"),
              Output("segment_sum", "children"),
              Output("service_sum", "children"),
              Input("url","pathname")
)
def update_card1(n):    
    ner_table = pd.read_csv(r"data\preprocess\FinancialPhraseBank-v1.0\ner_table.csv")
    
    auto_sum = int(ner_table[ner_table["Entities"]=="AUTO"]["Sum"])
    cc_service_sum = int(ner_table[ner_table["Entities"]=="CC SERVICE"]["Sum"])
    fd_sum = int(ner_table[ner_table["Entities"]=="FD"]["Sum"])
    investment_sum = int(ner_table[ner_table["Entities"]=="INVESTMENT"]["Sum"])
    mortgage_sum = int(ner_table[ner_table["Entities"]=="MORTGAGE"]["Sum"])
    new_cc_sum = int(ner_table[ner_table["Entities"]=="NEW CC"]["Sum"])
    segment_sum = int(ner_table[ner_table["Entities"]=="SEGMENT"]["Sum"])
    service_sum = int(ner_table[ner_table["Entities"]=="SERVICE"]["Sum"])
    
    total_queries = auto_sum + cc_service_sum + fd_sum + investment_sum + mortgage_sum + new_cc_sum + segment_sum + service_sum

    return "{}".format(total_queries), "{}".format(auto_sum), "{}".format(cc_service_sum), "{}".format(fd_sum), "{}".format(investment_sum), "{}".format(mortgage_sum), "{}".format(new_cc_sum), "{}".format(segment_sum), "{}".format(service_sum)


@app.callback(Output("entities-periods", "figure"), 
              Input("entities-checklist", "value"))
def update_sentiments_period(value):
    ner_table = pd.read_csv(r"data\preprocess\FinancialPhraseBank-v1.0\ner_table.csv", index_col=[0])
    mask = ner_table["Entities"].isin(value)
    fig = px.imshow(ner_table[mask].set_index(["Entities"]).drop(["Sum"], axis=1),
                    labels=dict(x="Periods", y="Products"),
                    color_continuous_scale="tealrose",
                    aspect="auto",
                    text_auto=True)
    fig.update_xaxes(side="top")
    fig.layout.height = 500
    fig.layout.width = 1000

    return fig





