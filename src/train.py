import os
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import Config
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

import warnings
warnings.filterwarnings("ignore")

fname = os.path.join(Config.FILES["RAW_DATA_DIR"], Config.FILES["RAW_INTENTS_FILE"])
with open("{}.json".format(fname), "r") as f:
    intents = json.load(f)
    
all_words = []
tags = []
xy = []
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
        
ignore_words = ["?", "!", ".", ",", ":"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label) # cross entropy loss needs only class labels, not one-hot
    
X_train = np.array(X_train)
y_train = np.array(y_train)

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = Config.MODELLING_CONFIG["HIDDEN_SIZE"]
output_size = len(tags)
# print(input_size, output_size)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
        
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    
    def __len__(self):
        return self.n_samples
    
    
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, 
                          batch_size=Config.MODELLING_CONFIG["BATCH_SIZE"],
                          shuffle=Config.MODELLING_CONFIG["SHUFFLE"],
                          num_workers=Config.MODELLING_CONFIG["NUM_WORKERS"],)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(Config.MODELLING_CONFIG["NUM_EPOCHS"]):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # # forward
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # # backward & optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print("Epoch [{}/{}], Loss:{:.4f}".format((epoch+1), Config.MODELLING_CONFIG["NUM_EPOCHS"], loss.item()))

print(f"final loss: loss={loss.item():.4f}")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

MODEL_FILE = os.path.join(Config.FILES["MODEL_DATA_DIR"], "chatbot.pth")
torch.save(data, MODEL_FILE)

print("training complete, file saved to {}".format(MODEL_FILE))