import os
import pandas as pd
import time as time_pck
import sd_material_ui as sdmui
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from datetime import datetime, time, timedelta

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

financial_df = pd.read_csv("data/preprocess/FinancialPhraseBank-v1.0/process_financial_phrase_data.csv")
sub_financial_df = financial_df[["sentence", "sentiment"]]
all_sentences = " ".join([str(text) for text in financial_df["sentence_processed"]])

header = [
    html.Thead(
        html.Tr(
            [
                html.Th("Sentences"),
                html.Th("Sentiments"),
            ]
        )
    )
]
row1 = html.Tr([html.Td(sub_financial_df["sentence"][0]), html.Td(sub_financial_df["sentiment"][0])])
row2 = html.Tr([html.Td(sub_financial_df["sentence"][1]), html.Td(sub_financial_df["sentiment"][1])])
row3 = html.Tr([html.Td(sub_financial_df["sentence"][2]), html.Td(sub_financial_df["sentiment"][2])])
row4 = html.Tr([html.Td(sub_financial_df["sentence"][3]), html.Td(sub_financial_df["sentiment"][3])])
row5 = html.Tr([html.Td(sub_financial_df["sentence"][4]), html.Td(sub_financial_df["sentiment"][4])])
row6 = html.Tr([html.Td(sub_financial_df["sentence"][5]), html.Td(sub_financial_df["sentiment"][5])])
row7 = html.Tr([html.Td(sub_financial_df["sentence"][6]), html.Td(sub_financial_df["sentiment"][6])])
row8 = html.Tr([html.Td(sub_financial_df["sentence"][7]), html.Td(sub_financial_df["sentiment"][7])])
row9 = html.Tr([html.Td(sub_financial_df["sentence"][8]), html.Td(sub_financial_df["sentiment"][8])])
row10 = html.Tr([html.Td(sub_financial_df["sentence"][9]), html.Td(sub_financial_df["sentiment"][9])])
body = [html.Tbody([row1, row2, row3, row4, row5, row5, row7, row8])]


def plotly_wordcloud(text):
    wc = WordCloud(stopwords=set(STOPWORDS),
                   max_words=300,
                   max_font_size=100)
    wc.generate(text)
    
    word_list=[]
    freq_list=[]
    fontsize_list=[]
    position_list=[]
    orientation_list=[]
    color_list=[]

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)
        
    # get the positions
    x=[]
    y=[]
    for i in position_list:
        x.append(i[0])
        y.append(i[1])
            
    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i*100)
    new_freq_list
    
    trace = go.Scatter(x=x, 
                       y=y, 
                       textfont = dict(size=new_freq_list,
                                       color=color_list),
                       hoverinfo="text",
                       hovertext=["{0}{1}".format(w, f) for w, f in zip(word_list, freq_list)],
                       mode="text",  
                       text=word_list
                      )
    
    layout = go.Layout({"xaxis": {"showgrid": False, "showticklabels": False, "zeroline": False},
                        "yaxis": {"showgrid": False, "showticklabels": False, "zeroline": False}},
                       width=1000,
                       height=400,
                       margin=dict(l=140, r=40, t=20, b=20))
    
    fig = go.Figure(data=[trace], layout=layout)
    
    return fig


layout = html.Div(
    style={"margin_top":"70px"},
    children=[
        dmc.Group(
            direction="column",
            grow=True,
            children=[
                dmc.Title(children="Customer Base", order=3, style={'text-align':'center', 'color' :'slategray'}),
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
                                        dmc.Text(id="totalcust", size="xl", weight=700),
                                        dmc.Text('Customer Historic Chats Records', id="churn_rate", size="xs", color='red')
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
                                        children=[DashIconify(icon="mdi:emoticon-happy", width=30)]
                                    )
                                ),
                                dmc.Group(
                                    direction='column',
                                    position='center',
                                    spacing='xs',
                                    style={'margin-top':10},
                                    children=[
                                        dmc.Text('Positive Sentences', size='xs', color='dimmed'),
                                        dmc.Text(id="positive", size="xl", weight=700),
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
                                        children=[DashIconify(icon="material-symbols:sentiment-neutral", width=30)]
                                    )
                                ),
                                dmc.Group(
                                    direction='column',
                                    position='center',
                                    spacing='xs',
                                    style={'margin-top':10},
                                    children=[
                                        dmc.Text('Neutral Sentences', size='xs', color='dimmed'),
                                        dmc.Text(id="neutral", size="xl", weight=700),
                                        
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
                                        children=[DashIconify(icon="solar:sad-circle-bold", width=30)]
                                    )
                                ),
                                dmc.Group(
                                    direction='column',
                                    position='center',
                                    spacing='xs',
                                    style={'margin-top':10},
                                    children=[
                                        dmc.Text('Negative Sentences', size='xs', color='dimmed'),
                                        dmc.Text(id="negative", size="xl", weight=700),
                                    ]
                                )
                            ],
                        ),
                    ]
                ),
                dmc.Divider(label="Sentiments Analysis", labelPosition="center", size="xl"),
                dmc.Group(
                    direction="row",
                    grow=True,
                    children=[
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"500px"},
                            children=[
                                dmc.Title("Sentences Sentiments", order=4, style={'text-align':'center', 'color':'grey'}),
                                dcc.Dropdown(
                                    id="periods-checklist",
                                    options=[
                                        {"label": "positive", "value": "positive"},
                                        {"label": "neutral", "value": "neutral"},
                                        {"label": "negative", "value": "negative"},
                                    ],
                                    value=["positive", "neutral", "negative"],
                                    multi=True,
                                    ),
                                dcc.Graph(id="sentiments-periods"),
                                ]
                            ),
                        ]
                    ),
                dmc.Divider(label="Sentences Analysis", labelPosition="center", size="xl"),
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
                                dmc.Title("Sentences Length", order=4, style={'text-align':'center', 'color':'grey'}),
                                dcc.Graph(id="sentiments-lengths"),
                                dcc.Checklist(
                                    id="lengths-checklist",
                                    options=["negative", "neutral", "positive"],
                                    value=["negative", "neutral", "positive"],
                                    inline=True
                                    ),
                                ]
                            ),
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"600px"},
                            children=[
                                dmc.Title("Sentences Data", order=4, style={'text-align':'center', 'color':'grey'}),
                                dmc.Table(header + body),
                                ]
                            ),
                        ]
                    ),
                dmc.Divider(label="Words Analysis", labelPosition="center", size="xl"),
                dmc.Group(
                    direction="row",
                    grow=True,
                    children=[
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            shadow="xs",
                            p="sm",
                            style={"height":"500px"},
                            children=[
                                dmc.Title("Word Clouds", order=4, style={'text-align':'center', 'color':'grey'}),
                                dcc.Graph(figure=plotly_wordcloud(text=all_sentences)),
                                ]
                            ),
                        ]
                    ),
                dmc.Space(h=100)
            ]
        )
    ],
)

@app.callback(Output("totalcust", "children"),
              Output("positive", "children"),
              Output("neutral", "children"),
              Output("negative", "children"),
              Input("url","pathname")
)
def update_card1(n):
    sentiments_df = pd.read_csv("data/preprocess/FinancialPhraseBank-v1.0/process_financial_phrase_data.csv")
    positive = sentiments_df.query("sentiment=='positive'")
    neutral = sentiments_df.query("sentiment=='neutral'")
    negative = sentiments_df.query("sentiment=='negative'")

    return "{}".format(sentiments_df["sentence"].nunique()), "{}".format(positive.shape[0]), "{}".format(neutral.shape[0]), "{}".format(negative.shape[0])


@app.callback(Output("sentiments-periods", "figure"), 
              Input("periods-checklist", "value"))
def update_sentiments_period(value):
    sentiments_df = pd.read_csv("data/preprocess/FinancialPhraseBank-v1.0/sentiments_periods.csv")
    mask = sentiments_df["sentiment"].isin(value)
    fig = px.line(sentiments_df[mask], x="month", y="count", color="sentiment", title="Sentences Sentiments Across Periods")
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Count")

    return fig


@app.callback(Output("sentiments-lengths", "figure"), 
              Input("lengths-checklist", "value"))
def update_reviews_length(value):
    sentiments_df = pd.read_csv("data/preprocess/FinancialPhraseBank-v1.0/process_financial_phrase_data.csv")
    mask = sentiments_df["sentiment"].isin(value)
    fig = px.histogram(sentiments_df[mask], x="length", color="sentiment", marginal="rug", hover_data=sentiments_df.columns, title="")
    fig.update_xaxes(title_text="Sentences Length")
    fig.update_yaxes(title_text="Density")
    
    return fig


