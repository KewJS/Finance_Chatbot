import os
import json
import random

import torch

from config import Config
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

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