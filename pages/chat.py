import os
import time
import json
import torch
import random

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc, dash_table, MATCH, ALL, ctx
# from transformers import AutoModelWithLMHead, AutoTokenizer

from app import app
from src.config import Config
from src.train.feedforward import NeuralNet
from src.preprocess.nltk_utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fname = os.path.join(Config.FILES["RAW_DATA_DIR"], Config.FILES["RAW_INTENTS_FILE"])
with open("{}.json".format(fname), "r") as f:
    intents = json.load(f)
    
MODEL_FILE = os.path.join(Config.FILES["MODEL_DATA_DIR"], "chatbot.pth")

model_data = torch.load(MODEL_FILE)
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data["all_words"]
tags = model_data["tags"]
model_state = model_data["model_state"]

model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_response(sentence):
    # print("Let's chat, type 'quit' to exit.")
    # while True:
    #     sentence = input("You: ")
    #     if sentence == "quit":
    #         print("Bye bye, hope to see you again very soon!")
    #         break
        
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
                # print("{}: {}".format(Config.MODELLING_CONFIG["BOT_NAME"], random.choice(intent["responses"])))
    # else:
        # print("{}: I do not understand this yet...".format(Config.MODELLING_CONFIG["BOT_NAME"]))
        
    return "Sorry, I am still learning and do not understand this..."

# # device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# print(f"Device: {device}")

# print("Start loading model...")
# name = "microsoft/DialoGPT-medium"
# tokenizer = AutoTokenizer.from_pretrained(name)
# model = AutoModelWithLMHead.from_pretrained(name)

# # Switch to cuda, eval mode, and FP16 for faster inference
# if device == "cuda":
#     model = model.half()
# model.to(device)
# model.eval()

# print("Done.")


def textbox(text, box="other"):
    style = {
        "max-width": "55%",
        "width": "max-content",
        "padding": "10px 15px",
        "border-radius": "25px",
    }

    if box == "self":
        style["margin-left"] = "auto"
        style["margin-right"] = 0
        color = "red"
        inverse = True

    elif box == "other":
        style["margin-left"] = 0
        style["margin-right"] = "auto"
        color = "light"
        inverse = False

    else:
        raise ValueError("Incorrect option for `box`.")

    return dbc.Card(text, style=style, body=True, color=color, inverse=inverse)


conversation = html.Div(
    style={
        "width": "80%",
        "max-width": "800px",
        "height": "70vh",
        "margin": "auto",
        "overflow-y": "auto",
    },
    id="display-conversation",
)

controls = dbc.InputGroup(
    style={"width": "80%", "max-width": "800px", "margin": "auto"},
    children=[
        dbc.Input(id="user-input", placeholder="Send a message", type="text"),
        dbc.InputGroup(dbc.Button("Submit", id="submit", color="danger")),
    ],
)


# # Define app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# server = app.server


# Define Layout
layout = dbc.Container(
    fluid=True,
    children=[
        html.H1("RAK-Voice Chatbot"),
        html.Hr(),
        dcc.Store(id="store-conversation", data=""),
        conversation,
        controls,
    ],
)


@app.callback(
    Output("display-conversation", "children"), [Input("store-conversation", "data")]
)
def update_display(chat_history):
    return [
        textbox(x, box="self") if i % 2 != 0 else textbox(x, box="other") for i, x in enumerate(chat_history)
    ]


@app.callback(
    [Output("store-conversation", "data"), Output("user-input", "value")],
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
    [State("user-input", "value"), State("store-conversation", "data")],
)
def run_chatbot(n_clicks, n_submit, user_input, chat_history):
    if n_clicks == 0:
        return "", ""

    if user_input is None or user_input == "":
        return [chat_history, ""]
    
    # # temporary
    # return chat_history + user_input + "<|endoftext|>" + user_input + "<|endoftext|>", ""

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    # bot_input_ids = tokenizer.encode(
    #     chat_history + user_input + tokenizer.eos_token, return_tensors="pt"
    # ).to(device)
    
    # generated a response while limiting the total chat history to 1000 tokens,
    # chat_history_ids = model.generate(
    #     bot_input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id
    # )
    # chat_history = tokenizer.decode(chat_history_ids[0])
    
    chat = []
    chat.append(user_input)
    text = get_response(user_input)
    chat.append(text)

    return chat, ""