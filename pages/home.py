import os
import base64
import pandas as pd
import time as time_pck
import sd_material_ui as sdmui
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time, timedelta

import dash
import dash_daq as daq
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc, dash_table, MATCH, ALL, ctx

from app import app

dash.register_page(__name__, path="/")

def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


layout = dbc.Container([
    html.H1("Welcome to RAK-Chat Language System"),
    html.Hr(),
    html.Br(),
    
    dbc.Card([
        dbc.CardBody([
            html.Div("RAK-Chat: Intelligent Virtual Assistant",
                     style={"fontsize": 100, "font-weight": "bold", "text_align": "center"}),
            html.Br(),
            dbc.Row([
                html.Div(
                    """
                    To improve customers users experiences on digital platform, to reduce the reaction time between customers and RAKbank digital platform, we invent a smart digital chatbot that servces customers on RAKbank products. This chatbot will be used to facilitate conversation with customers and RAKbank. 
                    """, className="mt-2"
                ),
                html.Br(),
                html.Div(
                    """
                    RAK-Chat Virtual Assistance is built using Generative Model, Gnerative Pre-Trained Transformer. State-of-Art large language model that will consume queries from customers and generate the most suitable responses. 
                    """, className="mt-2"
                ),
                html.Br(),
                html.Hr(),
                html.Img(src=b64_image("assets/chatbot.png"), style={"height":"30%", "width":"60%", "textAlign": "center", 'display': 'block', 'margin': 'auto'}),
                html.Br(),
            ]),   
        ]),
    ]),
    
    dbc.Card([
        dbc.CardBody([
            html.Div("RAK-Chat: Products Entities Summarization",
                     style={"fontsize": 100, "font-weight": "bold", "text_align": "center"}),
            html.Br(),
            dbc.Row([
                html.Div(
                    """
                    To improve RAKbank to keep up with latest products trend from customers from time to time, queries from customers on products will be essential to understand what kind of products customers are looking for at the moment. By keeping with latest users products trend, RAKbank will be able to react to provides the best products to customers at the right time. 
                    """, className="mt-2"
                ),
                html.Br(),
                html.Div(
                    """
                    Products Entities Summarization was built using Convolutional Neural Network (CNN) model to capture the key words presence in customers queries. From there, customers need from time to time can be captured.
                    """, className="mt-2"
                ),
                html.Br(),
                html.Hr(),
                html.Img(src=b64_image("assets/ner.png"), style={"height":"30%", "width":"60%", "textAlign": "center", 'display': 'block', 'margin': 'auto'}),
                html.Br(),
            ]),
        ]),
    ]),
    
    dbc.Card([
        dbc.CardBody([
            html.Div("RAK-Chat: Sentiments Analysis",
                     style={"fontsize": 100, "font-weight": "bold", "text_align": "center"}),
            html.Br(),
            dbc.Row([
                html.Div(
                    """
                    To improve RAKbank customer services by understanding customers satisfaction from customers queries. From customers satisfaction, we will be able to understand where to improve, hence, build better customer exepriences for customers.
                    """, className="mt-2"
                ),
                html.Br(),
                html.Div(
                    """
                    Sentiment Analysis built using <code>Generative large language model, BERT</code>, based on concept of Transformer. Feelings from customers queries can be predicted from time to time.
                    """, className="mt-2"
                ),
                html.Br(),
                html.Hr(),
                html.Img(src=b64_image("assets/sentiment.png"), style={"height":"30%", "width":"60%", "textAlign": "center", 'display': 'block', 'margin': 'auto'}),
                html.Br(),
            ]),
        ]),
    ]),
], className="mt-4")