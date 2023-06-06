<<<<<<< HEAD
import os
import json
import time
import pickle
import numpy as np
from tqdm import tqdm
from typing import List
from collections import Counter
import nltk
from nltk.stem.porter import PorterStemmer

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from src.config import Config
# from src.preprocess.nltk_utils import NLTK_Utils

STEMMER = PorterStemmer()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")

class Logger():
    info = print
    warnings = print
    critical = print
    error = print

class Preprocess(Config):
    data = {}
    
    def __init__(self, suffix="", logger=Logger()):
        self.suffix = suffix
        self.logger = logger
        
        
    def get_personachat_data(self):
        start = time.time()
        self.logger.info("Read Persona Chat Data:")
        fname = os.path.join(self.FILES["RAW_DATA_DIR"], "personachat_self_original")
        with open("{}.json".format(fname), "r") as f:
            self.data["persona_chat"] = json.load(f)
            
        self.data["dataset"] = self.tokenize(self.data["persona_chat"])
        self.personalities = [dialog['personality'] for dialog in self.data["dataset"]["train"]]
        
        if self.QDEBUG:
            fname = os.path.join(self.FILES["RAW_PREPROCESS_DIRDATA_DIR"], "personachat_self_cached")
            with open(fname, "wb") as fp:
                pickle.dump(self.data["dataset"], fp)
            
            fname = os.path.join(self.FILES["RAW_PREPROCESS_DIRDATA_DIR"], "personachat_self_personalities")
            with open(fname, 'wb') as fp:
                pickle.dump(self.personalities, fp)


    def get_chat_data(self) -> None:
        start = time.time()
        self.logger.info("Read Conversational Data:")
        fname = os.path.join(self.FILES["RAW_DATA_DIR"], self.FILES["RAW_INTENTS_FILE"])
        with open("{}.json".format(fname), "r") as f:
            self.data["intents"] = json.load(f)

        self.logger.info("  tokenize all the words...")
        self.all_words = []
        self.tags = []
        self.xy = []
        for intent in self.data["intents"]:
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


    def get_counter_and_lens(self, data, tokenizer):
        flatten = lambda l: [item for sublist in l for item in sublist]
        toks = [tokenizer.tokenize(" ".join(x)) for x in data] # [tokenizer.tokenize(x) for x in data]
        
        return list(map(len, toks)), Counter(flatten(toks)), Counter(" ".join(flatten(data)).split())


    def plot_counts(self, counts, top_k=30):
        labels, values = zip(*counts.most_common()[:top_k])
        indexes = np.arange(len(labels))
        width = 1
        plt.figure(num=None, figsize=(28, 7), dpi=60, facecolor="w", edgecolor="k")
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.show()
        
        
    def plot_hist(self, lens, n_bins=50):
        n, bins, patches = plt.hist(lens, n_bins, facecolor="blue", alpha=0.9)
        plt.show()
        
        
    def stem(self, word):
        return STEMMER.stem(word.lower())
    
        
    # def tokenize(self, sentence):
    #     return nltk.word_tokenize(sentence)


    def bag_of_words(self, tokenized_sentence, words):
        """
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag = [0, 1, 0, 1, 0, 0, 0]
        """
        sentence_words = [self.stem(word) for word in tokenized_sentence]
        bag = np.zeros(len(words), dtype=np.float32)
        
        for idx, w in enumerate(words):
            if w in sentence_words:
                bag[idx] = 1
                
        return bag
        
      
    def tokenize(self, obj):
        if isinstance(obj, list):
            return [self.tokenize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: self.tokenize(v) for k, v in obj.items()}
        
        return TOKENIZER.encode(obj)
        
        
class CustomDataset(Dataset):
    def __init__(self, args, tokenizer, data_type):
        assert data_type in ["train", "valid", "test"]
        
        print(f"Loading {data_type} data...")
        with open(f"{args.task_dir}/{data_type}.pickle", "rb") as f:
            dials = pickle.load(f)
        
        
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
    
        
=======
import os
import json
import time
import pickle
import numpy as np
from tqdm import tqdm
from typing import List
from collections import Counter
import nltk
from nltk.stem.porter import PorterStemmer

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from src.config import Config
# from src.preprocess.nltk_utils import NLTK_Utils

STEMMER = PorterStemmer()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")

class Logger():
    info = print
    warnings = print
    critical = print
    error = print

class Preprocess(Config):
    data = {}
    
    def __init__(self, suffix="", logger=Logger()):
        self.suffix = suffix
        self.logger = logger
        
        
    def get_personachat_data(self):
        start = time.time()
        self.logger.info("Read Persona Chat Data:")
        fname = os.path.join(self.FILES["RAW_DATA_DIR"], "personachat_self_original")
        with open("{}.json".format(fname), "r") as f:
            self.data["persona_chat"] = json.load(f)
            
        self.data["dataset"] = self.tokenize(self.data["persona_chat"])
        self.personalities = [dialog['personality'] for dialog in self.data["dataset"]["train"]]
        
        if self.QDEBUG:
            fname = os.path.join(self.FILES["RAW_PREPROCESS_DIRDATA_DIR"], "personachat_self_cached")
            with open(fname, "wb") as fp:
                pickle.dump(self.data["dataset"], fp)
            
            fname = os.path.join(self.FILES["RAW_PREPROCESS_DIRDATA_DIR"], "personachat_self_personalities")
            with open(fname, 'wb') as fp:
                pickle.dump(self.personalities, fp)


    def get_chat_data(self) -> None:
        start = time.time()
        self.logger.info("Read Conversational Data:")
        fname = os.path.join(self.FILES["RAW_DATA_DIR"], self.FILES["RAW_INTENTS_FILE"])
        with open("{}.json".format(fname), "r") as f:
            self.data["intents"] = json.load(f)

        self.logger.info("  tokenize all the words...")
        self.all_words = []
        self.tags = []
        self.xy = []
        for intent in self.data["intents"]:
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


    def get_counter_and_lens(self, data, tokenizer):
        flatten = lambda l: [item for sublist in l for item in sublist]
        toks = [tokenizer.tokenize(" ".join(x)) for x in data] # [tokenizer.tokenize(x) for x in data]
        
        return list(map(len, toks)), Counter(flatten(toks)), Counter(" ".join(flatten(data)).split())


    def plot_counts(self, counts, top_k=30):
        labels, values = zip(*counts.most_common()[:top_k])
        indexes = np.arange(len(labels))
        width = 1
        plt.figure(num=None, figsize=(28, 7), dpi=60, facecolor="w", edgecolor="k")
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.show()
        
        
    def plot_hist(self, lens, n_bins=50):
        n, bins, patches = plt.hist(lens, n_bins, facecolor="blue", alpha=0.9)
        plt.show()
        
        
    def stem(self, word):
        return STEMMER.stem(word.lower())
    
        
    # def tokenize(self, sentence):
    #     return nltk.word_tokenize(sentence)


    def bag_of_words(self, tokenized_sentence, words):
        """
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag = [0, 1, 0, 1, 0, 0, 0]
        """
        sentence_words = [self.stem(word) for word in tokenized_sentence]
        bag = np.zeros(len(words), dtype=np.float32)
        
        for idx, w in enumerate(words):
            if w in sentence_words:
                bag[idx] = 1
                
        return bag
        
      
    def tokenize(self, obj):
        if isinstance(obj, list):
            return [self.tokenize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: self.tokenize(v) for k, v in obj.items()}
        
        return TOKENIZER.encode(obj)
        
        
class CustomDataset(Dataset):
    def __init__(self, args, tokenizer, data_type):
        assert data_type in ["train", "valid", "test"]
        
        print(f"Loading {data_type} data...")
        with open(f"{args.task_dir}/{data_type}.pickle", "rb") as f:
            dials = pickle.load(f)
        
        
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
    
        
>>>>>>> 9ecf694929e32c4fc0269d6cb0bc3b2d6a6766e6
