import os
import json
import time
import pickle
import numpy as np
from typing import List
from collections import Counter

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset

from src.config import Config
from src.preprocess.nltk_utils import NLTK_Utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger():
    info = print
    warnings = print
    critical = print
    error = print

class Preprocess(NLTK_Utils):
    data = {}
    
    def __init__(self, suffix="", logger=Logger()):
        self.suffix = suffix
        self.logger = logger
        

    def get_chat_data(self) -> None:
        start = time.time()
        self.logger.info("Read Conversational Data:")
        fname = os.path.join(Config.FILES["RAW_DATA_DIR"], Config.FILES["RAW_INTENTS_FILE"])
        with open("{}.json".format(fname), "r") as f:
            self.data["intents"] = json.load(f)

        self.logger.info("  tokenize all the words...")
        self.all_words = []
        self.tags = []
        self.xy = []
        for intent in self.data["intents"]["intents"]:
            tag = intent["tag"]
            self.tags.append(tag)
            for pattern in intent["patterns"]:
                w = self.tokenize(pattern)
                self.all_words.extend(w)
                self.xy.append((w, tag))
        
        self.logger.info("  stemming the words...")
        ignore_words = ["?", "!", ".", ",", ":"]
        self.all_words = [self.stem(w) for w in self.all_words if w not in self.ANALYSIS_CONFIG["IGNORE_CHARACTERS"]]
        self.all_words = sorted(set(self.all_words))
        self.tags = sorted(set(self.tags))

        self.logger.info("  create train & test data...")
        self.X_train = []
        self.y_train = []
        for (pattern_sentence, tag) in self.xy:
            bag = self.bag_of_words(pattern_sentence, self.all_words)
            self.X_train.append(bag)
            
            label = self.tags.index(tag)
            self.y_train.append(label) # cross entropy loss needs only class labels, not one-hot
            
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        if self.QDEBUG:
            self.logger.info("  export the data into {}...".format(self.FILES["PREPROCESS_DIR"]))
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "{}.txt".format(self.FILES["TRAIN_INTENTS_FILE"]))
            np.savetxt(fname, self.X_train, fmt="%s")
                
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "{}.txt".format(self.FILES["TEST_INTENTS_FILE"]))
            np.savetxt(fname, self.y_train, fmt="%s")
                
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "{}.pickle".format(self.FILES["ALL_WORDS_LIST"]))
            fname = open(fname, "wb")
            pickle.dump(self.all_words, fname)
            
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "{}.pickle".format(self.FILES["TAGS_LIST"]))
            fname = open(fname, "wb")
            pickle.dump(self.tags, fname)
            
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "{}.pickle".format(self.FILES["XY_LIST"]))
            fname = open(fname, "wb")
            pickle.dump(self.xy, fname)
            
        self.logger.info("  done creating X and Y data for conversational data using time: {:.2f}s".format(time.time()-start))
        
        
    def __repr__(self):
        return "Size of X-train data: {}, Size of y-train data: {}".format(self.X_train.shape, self.y_train.shape)
        
        
class ChatDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x_data = x_train
        self.y_data = y_train
        self.n_samples = len(x_train)
        
        
    def __getitem__(self, index:int) -> List[int]:
        return self.x_data[index], self.y_data[index]
    
    
    def __len__(self):
        return self.n_samples
    
    
    def create_masks(self, input, target):
        def subsequent_mask(size):
            mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
            return mask.unsqueeze(0)
        
        question_mask = (question!=0).to(DEVICE)
        question_mask = question_mask.unsqueeze(1).unsqueeze(1)
        
        input_mask = input_mask!=0
        input_mask = input_mask.unsqueeze(1)
        input_mask = input_mask & subsequent_mask(input.size(-1)).type_as(input_mask.data)
        input_mask = input_mask.unsqueeze(1)
        target_mask = target_mask!=0
        
        return question_mask, input_mask, target_mask
    
    
class AdamWarmup(Config):
    def __init__(self, model_size, warmup_steps, optimizer):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0
        
        
    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))
    
    
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.lr = lr
        self.optimizer.step()
        
        
class LossWithLS(nn.Module):
    def __init__(self, size, smooth):
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size
        
    
    def forward(self, prediction, target, mask):
        """Prediction of shape, (batch_size, max_words, vocab_size)
        
        Target and mask of shape: (batch_size, max_words)

        Args:
            prediction (_type_): _description_
            target (_type_): _description_
            mask (_type_): _description_
        """
        prediction = prediction.view(-1, prediction.size(-1))
        target = target.contiguous().view(-1)
        mask = mask.float()
        mask = mask.view(-1)
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size-1))
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = self.criterion(prediction, labels)
        loss = (loss.sum(1) * mask).sum() / mask.sum()
        
        return loss
            
            
if __name__ == "__main__":
    preprocess = Preprocess()
    preprocess.get_chat_data()