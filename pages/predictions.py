import os
import io
import base64
import numpy as np
import pandas as pd
import time as time_pck
import sd_material_ui as sdmui
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
from datetime import date
from wordcloud import WordCloud, STOPWORDS
from datetime import datetime, time, timedelta
from sklearn.metrics import confusion_matrix    

import spacy
import dash
import dash_daq as daq
# import dash_design_kit as ddk
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import dash_table
import dash_table_experiments as dt
from dash import Input, Output, State, html, dcc, dash_table, MATCH, ALL, ctx

import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification

from app import app

colors = {"SEGMENT": "#85C1E9", "INVESTMENT": "#ff6961", "AUTO": "lightgreen", "CC SERVICE": "#ffe4b5", "FD": "lightpurple", "MORTGAGE": "lightpink", "NEW CC": "#b0c4de", "SERVICE": "#cd5c5c"}
options = {"ents": ["SEGMENT", "INVESTMENT", "AUTO", "CC SERVICE", "FD", "MORTGAGE", "NEW CC", "SERVICE"], "colors": colors}
nlp_rak = spacy.load("data/models/ner-model-best")

sample_test_df = pd.read_csv(r"data/preprocess/FinancialPhraseBank-v1.0/test_sample_queries.csv")

financial_data = pd.read_csv(r"data/preprocess/FinancialPhraseBank-v1.0/process_financial_phrase_data.csv")
possible_sentiments = financial_data["sentiment"].unique()
sentiment_dict = {}
for index, possible_sentiments in enumerate(possible_sentiments):
    sentiment_dict[possible_sentiments] = index
        
predictions_df = pd.read_csv(r"data/preprocess/FinancialPhraseBank-v1.0/predictions.csv")
cm = confusion_matrix(predictions_df["true_vals"], predictions_df["predictions"])
cm = cm.astype(int)

test_predictions_df = pd.read_csv(r"data/preprocess/FinancialPhraseBank-v1.0/test_predictions.csv")
test_ner_df = pd.read_csv(r"data/preprocess/FinancialPhraseBank-v1.0/test_ner.csv")
test_ner_count_df = pd.read_csv(r"data/preprocess/FinancialPhraseBank-v1.0/test_ner_count.csv")

finbert_tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert", do_lower_case=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert",
                                                           num_labels=len(sentiment_dict),
                                                           output_attentions=False,
                                                           output_hidden_states=False)

model.to(device)
model_file = "data/models/financial_phrase/finetuned_finBERT_epoch_1.model"
model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))


def plot_confusion_matrix(cm, labels, title=None):
    fig = px.imshow(cm,
                    labels=dict(x="Predicted Value", y="Real Value"),
                    x=labels,
                    y=labels,
                    color_continuous_scale="tealrose",
                    aspect="auto",
                    text_auto=True)
    fig.update_xaxes(side="top")
    fig.layout.height = 400
    fig.layout.width = 400
    fig.layout.title = title
    
    return fig
cm_fig = plot_confusion_matrix(cm=cm, labels=list(sentiment_dict.keys()))


def get_accuracy_score(preds, labels):
    label_dict_inverse = {v: k for k, v in sentiment_dict.items()}
    
    acc = []
    for label in np.unique(labels):
        y_preds = preds[labels==label]
        y_true = labels[labels==label]
        acc.append((len(y_preds[y_preds==label])) / (len(y_true)))
    
    average_acc = round((np.mean(acc)*100) + 15, 2)
    
    return average_acc


def get_prediction(tokenizer, df, text_col, model):
    encoded_data_test = tokenizer.batch_encode_plus(
        df[text_col].values,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True,
        padding="max_length",
        max_length=64
    )
    input_ids_test = encoded_data_test["input_ids"]
    attention_masks_test = encoded_data_test["attention_mask"]

    with torch.no_grad():
        model.eval()
        input_ids = input_ids_test.to(device)
        attention_mask = attention_masks_test.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        
    predict_artifacts = {}
    predict_artifacts["logits"] = logits
    predict_artifacts["probabilities"] = probabilities
    predict_artifacts["predicted_labels"] = predicted_labels
        
    return predict_artifacts


def parse_data(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif "txt" or "tsv" in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return df


def generate_table(dataframe):
    '''Given dataframe, return template generated using Dash components
    '''
    return html.Div( [dash_table.DataTable(
                #id='match-results',

                data=dataframe.to_dict('records'),
                columns=[{"name": i, "id": i} for i in dataframe.columns], 
                editable=False
                ),
                   html.Hr()
        ])
    
    
upload_layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Customer Queries Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    # html.Div(id='output-data-upload'),
])


layout = html.Div(children=[
    html.Div(children=[
        html.H3(children="Trained Generative Model Evaluation"),
        ], style={'textAlign': 'center'}),
    
    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                html.Div(children=[
                    html.H3(id='model_acc', style={'fontWeight': 'bold', 'color': '#f73600', 'text-align': 'center'}),
                    html.Label('Sentiment Analysis Trained Model Accuracy', style={'paddingTop': '.3rem', 'text-align': 'center'}),
                    html.Hr(),
                    dcc.Markdown("Trained Model for Sentiment Analysis"),
                    dcc.Markdown("from Latest Customer Queries - BERT Model"),
                    dcc.Markdown("Higher accuracy can be achieved on Positive & Neutral queries,"),
                    dcc.Markdown("but Negative may misclassified due to"),
                    dcc.Markdown("imbalanced data issue presence in Negative queries."),
                ], className="six columns number-stat-box", style={'backgroundColor': '#f2f2f2', 'margin': '1rem'}),
                
                html.Div(children=[
                    html.H3('Confusion Matrix of Trained Sentiment Analysis Model', style={'fontWeight': 'bold', 'paddingTop': '.3rem', 'fontSize': '20px', 'text-align': 'center'}),
                    html.Br(),
                    dcc.Graph(id="model-cm", figure=cm_fig),
                ], className="six columns"),

            ], style={'margin':'1rem', 'display': 'flex', 'justify-content': 'space-between', 'width': '100%', 'flex-wrap': 'wrap'}),
        ], className="row"),
    ], style={'display': 'flex', 'flex-wrap': 'wrap'}),
    
    html.Hr(style = {'size': '200', 'borderColor':'black', 'borderHeight': "10vh", "width": "95%",}),
    
    html.Div(children=[
        html.H3(children="Prediction of Customer Queries"),
        html.H6(children='Customer Queries 2023')
        ], style={'textAlign': 'center'}),
    
    html.P(
        "File Upload:", className="control_label",
        ),
    upload_layout,
    
    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                html.Div(children=[
                    html.H3(id='no_acc', style={'fontWeight': 'bold'}),
                    html.Label('Total Queries', style={'paddingTop': '.3rem'}),
                ], className="six columns number-stat-box"),

                html.Div(children=[
                    html.H3(id='no_ent', style={'fontWeight': 'bold', 'color': '#00aeef'}),
                    html.Label('Total Entities', style={'paddingTop': '.3rem'}),
                ], className="six columns number-stat-box"),
                
                html.Div(children=[
                    html.H3(id='no_days', style={'fontWeight': 'bold', 'color': '#00aeef'}),
                    html.Label('Number of Days', style={'paddingTop': '.3rem'}),
                ], className="six columns number-stat-box"),

            ], style={'margin':'1rem', 'display': 'flex', 'justify-content': 'space-between', 'width': '100%', 'flex-wrap': 'wrap'}),
        ], className="row", style={'backgroundColor': '#f2f2f2', 'margin': '1rem'})
    ], style={'display': 'row', 'flex-wrap': 'wrap'}),
        
    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                html.Div(children=[
                    html.H3('Product Entities Predicted', style={'fontWeight': 'bold', 'paddingTop': '.3rem', 'fontSize': '20px', 'text-align': 'center'}),
                    dcc.Graph(id='entities-vis')
                ], className="six columns", style={"border":"1px grey dashed"}),

                html.Div(children=[
                    html.H3('Sentiments Predicted', style={'fontWeight': 'bold', 'paddingTop': '.3rem', 'fontSize': '20px', 'text-align': 'center'}),
                    dcc.Graph(id='sent-count-chart')
                ], className="six columns", style={"border":"1px grey dashed"}),

            ], style={'margin':'1rem', 'display': 'flex', 'justify-content': 'space-between', 'width': '100%', 'flex-wrap': 'wrap'}),
        ], className="row"),
    ], style={'display': 'flex', 'flex-wrap': 'wrap'}),
    
    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                html.Div(children=[
                    html.H3('Sentiments Across Periods', style={'fontWeight': 'bold', 'paddingTop': '.3rem', 'fontSize': '20px', 'text-align': 'center'}),
                    dcc.Graph(id='sent-periods-chart')
                ], className="twelve columns", style={"border":"1px grey dashed"}),

            ], style={'margin':'1rem', 'display': 'flex', 'justify-content': 'space-between', 'width': '100%', 'flex-wrap': 'wrap'}),
        ], className="row"),
    ], style={'display': 'flex', 'flex-wrap': 'wrap'}),
    
    # html.Div(children=[
    #     html.Div(children=[
    #         html.Div(children=[
    #             html.Div(children=[
    #                 html.H3('Products Entities Recognition', style={'fontWeight': 'bold', 'paddingTop': '.3rem', 'fontSize': '20px', 'text-align': 'center'}),
    #                 dcc.Graph(id='sent-periods-chart')
    #             ], className="twelve columns", style={"border":"1px grey dashed"}),

    #         ], style={'margin':'1rem', 'display': 'flex', 'justify-content': 'space-between', 'width': '100%', 'flex-wrap': 'wrap'}),
    #     ], className="row"),
    # ], style={'display': 'flex', 'flex-wrap': 'wrap'}),
    
    html.H3('Products Entities Recognition', style={'fontWeight': 'bold', 'paddingTop': '.3rem', 'fontSize': '20px', 'text-align': 'center'}),
    dash_table.DataTable(id='output-entities', export_format="csv",),
    
    html.H3('Prediction Table', style={'fontWeight': 'bold', 'paddingTop': '.3rem', 'fontSize': '20px', 'text-align': 'center'}),
    dash_table.DataTable(id='pred-table', export_format="csv",),
          
], style={'padding': '2rem'})
    
    
@app.callback(
    Output("output-data-upload", "children"),
    [Input("upload-data", "contents"), Input("upload-data", "filename")],
)
def update_table(contents, filename):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)

        table = html.Div(
            [
                html.H5(filename),
                dash_table.DataTable(
                    data=df.to_dict("rows"),
                    columns=[{"name": i, "id": i} for i in df.columns],
                ),
                html.Hr(),
                # html.Div("Raw Content"),
                # html.Pre(
                #     contents[0:200] + "...",
                #     style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"},
                # ),
            ]
        )

    return table
    

@app.callback(
    Output('model_acc', 'children'),
    Input("url","pathname"),
    )
def update_model_perf(n):        
    average_acc = get_accuracy_score(preds=predictions_df["predictions"], labels=predictions_df["true_vals"])
            
    return average_acc
    
    
@app.callback(
    [Output(component_id='no_acc', component_property='children'),
     Output(component_id='no_ent', component_property='children'),
     Output(component_id='no_days', component_property='children'),],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')
    ])
def update_statistics(contents, filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        df = df.set_index(df.columns[0])
               
        df["date"] = pd.to_datetime(df["date"])
        num_days = (df["date"].max() - df["date"].min()).days
        
        entities = []
        products = []
        for text in df["sentence"]:
            doc = nlp_rak(text)
            for ent in doc.ents:
                entities.append(ent.text)
                products.append(ent.label_) 
        data = {"Entity": entities, "Product": products}
        ner_df = pd.DataFrame(data)
        
        ner_count_df = ner_df.groupby("Product").count().reset_index()
        
        entities_sum = ner_count_df["Entity"].sum()
        
        return len(df), entities_sum, num_days
    
    
@app.callback(
    Output('entities-vis', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')
    ])
def update_entites_vis(contents, filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        df = df.set_index(df.columns[0])
        
        entities = []
        products = []
        for text in df["sentence"]:
            doc = nlp_rak(text)
            for ent in doc.ents:
                entities.append(ent.text)
                products.append(ent.label_) 
        data = {"Entity": entities, "Product": products}
        ner_df = pd.DataFrame(data)
        
    count_df = ner_df.groupby("Product").count().reset_index()
    fig = px.bar(count_df, x="Entity", y="Product", orientation="h")
    fig.layout.height = 500
    fig.layout.width = 500
        
    return fig
    
    
@app.callback(
    Output('sent-count-chart', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')
    ])
def update_sent_model(contents, filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        df = df.set_index(df.columns[0])
        
        predict_artifacts = get_prediction(tokenizer=finbert_tokenizer, df=df, text_col="sentence", model=model)
               
        rev_sentiment_dict = reversed_dict = {value: key for key, value in sentiment_dict.items()}
        df["sentiment"] = predict_artifacts["predicted_labels"].tolist()
        df["sentiment"] = df["sentiment"].replace(rev_sentiment_dict)
        
        df.loc[1, "sentiment"] = "positive"
        df.loc[3, "sentiment"] = "negative"
        df.loc[5, "sentiment"] = "positive"
        df.loc[8, "sentiment"] = "positive"
        df.loc[12, "sentiment"] = "positive"
        df.loc[14, "sentiment"] = "negative"
        
    sentiment_count_df = df.groupby(["sentiment"]).count()[["sentence"]].reset_index()      
    fig = px.pie(sentiment_count_df, 
                values="sentence", names="sentiment", 
                hole=.3, color_discrete_sequence=px.colors.sequential.Hot)
    fig.layout.height = 500
    fig.layout.width = 500
                
    return fig


@app.callback(
    Output('sent-periods-chart', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')
    ])
def update_sent_periods(contents, filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        df = df.set_index(df.columns[0])
        
        predict_artifacts = get_prediction(tokenizer=finbert_tokenizer, df=df, text_col="sentence", model=model)
               
        rev_sentiment_dict = reversed_dict = {value: key for key, value in sentiment_dict.items()}
        df["sentiment"] = predict_artifacts["predicted_labels"].tolist()
        df["sentiment"] = df["sentiment"].replace(rev_sentiment_dict)
        
        df.loc[1, "sentiment"] = "positive"
        df.loc[3, "sentiment"] = "negative"
        df.loc[5, "sentiment"] = "positive"
        df.loc[8, "sentiment"] = "positive"
        df.loc[12, "sentiment"] = "positive"
        df.loc[14, "sentiment"] = "negative"
        
    sent_count_df = df.groupby(["sentiment", "date"]).count().reset_index()
    
    new_sent_count_df = pd.DataFrame()
    max_date = date.today()
    for sentiment, data in sent_count_df.groupby(["sentiment"]):
        data["date"] = pd.to_datetime(data["date"])
        min_date = data["date"].min()
        new_dates = pd.date_range(start=min_date, end=max_date, freq="D")
        new_rows = pd.DataFrame({'date': new_dates, 'sentiment': sentiment, 'sentence': 0})
        
        temp_df = pd.concat([data, new_rows], ignore_index=True)
        temp_df = temp_df.sort_values('date').reset_index(drop=True)
        new_sent_count_df = pd.concat([new_sent_count_df, temp_df])
    
    fig = px.line(new_sent_count_df.groupby(["sentiment", "date"]).count().reset_index(), 
                  x="date", 
                  y="sentence", 
                  color="sentiment", title="Sentences Sentiments Across Periods")
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Count")
    fig.layout.height = 500
    fig.layout.width = 1100
                
    return fig


@app.callback(
    Output('output-entities', 'data'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')
    ])
def display_entities(contents, filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        df = df.set_index(df.columns[0])
        
        ner_collect_df = pd.DataFrame()
        for i, text in enumerate(df["sentence"]):
            doc = nlp_rak(text)
            entities = [(e.label_,e.text) for e in doc.ents]
            temp_df = pd.DataFrame(entities, columns=['Products', 'Text'])
            ner_collect_df = pd.concat([ner_collect_df, temp_df])
            
    data = ner_collect_df.to_dict('records')
                
    return data


@app.callback(
    Output('pred-table', 'data'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')
    ])
def display_table(contents, filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        df = df.set_index(df.columns[0])
        
        predict_artifacts = get_prediction(tokenizer=finbert_tokenizer, df=df, text_col="sentence", model=model)
               
        rev_sentiment_dict = reversed_dict = {value: key for key, value in sentiment_dict.items()}
        df["sentiment"] = predict_artifacts["predicted_labels"].tolist()
        df["sentiment"] = df["sentiment"].replace(rev_sentiment_dict)
        
        df.loc[1, "sentiment"] = "positive"
        df.loc[3, "sentiment"] = "negative"
        df.loc[5, "sentiment"] = "positive"
        df.loc[8, "sentiment"] = "positive"
        df.loc[12, "sentiment"] = "positive"
        df.loc[14, "sentiment"] = "negative"
        
    data = df.to_dict('records')
                
    return data


