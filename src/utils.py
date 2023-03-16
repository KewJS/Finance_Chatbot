import json

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset

from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(Dataset):
    def __init__(self):
        fname = os.path.join(Config.FILES["RAW_DATA_DIR"], Config.FILES["RAW_INTENTS_FILE"])
        with open("{}.json".format(fname), "r") as f:
            self.pairs = json.load(f)
            
        self.dataset_size = len(self.pairs)
    
    
    def __getitem__(self, i):
        question = torch.long