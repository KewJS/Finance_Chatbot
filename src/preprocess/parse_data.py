import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from collections import Counter
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

from src.config import Args, Config

data_list = ["daily_dialog", "empathetic_dialogues", "persona_chat", "blended_skill_talk"]


class Logger():
    info = print
    warning = print
    error = print
    critical = print


class ParseData(Config):
    data = {}
    
    def __init__(self, logger=Logger(), args=Args()):
        self.logger = logger
        self.args = args
        
        
    def get_chat_data(self):
        self.parsed_dials, self.dial_sent_lens = self.parse_dialog(self.args)
        self.dialogues_lists = [d["turns"] for d in self.parsed_dials]
        
        self.logger.info("  convert RAK-Voice chat data into dataframe...")
        self.data["merged_df"] = pd.DataFrame(self.dialogues_lists)
        self.data["merged_df"] = self.data["merged_df"].add_prefix("context_")
        
        self.logger.info("Reading External Banking FaQ chat data:")
        self.data["ext_chat_df"] = pd.read_csv(os.path.join(self.FILES["RAW_DATA_DIR"], "{}.csv".format(self.FILES["RAW_BANK_FAQ_FILE"])))
        self.data["trans_ext_chat_df"] = pd.DataFrame(self.data["ext_chat_df"][["Question", "Answer"]].iloc[:1760].to_numpy().reshape(-1, 8))
        self.data["trans_ext_chat_df"] = self.data["trans_ext_chat_df"].add_prefix("context_")
        
        self.logger.info("  merging RAKbank chat data with external banking FaQ chat data...")
        self.data["merged_df"] = pd.concat([self.data["merged_df"], self.data["trans_ext_chat_df"]])
        self.data["merged_df"] = self.data["merged_df"].reset_index(drop=True)
        
        self.data["trn_df"], self.data["val_df"] = self.split_trn_val(df=self.data["merged_df"], args=self.args)
        
        if self.QDEBUG:
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "{}.parquet".format(self.FILES["PROCESS_RAKCHAT_FILE"]))
            self.data["merged_df"].to_parquet(fname)
            
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "{}.parquet".format(self.FILES["TRAIN_RAKCHAT_FILE"]))
            self.data["trn_df"].to_parquet(fname)
            
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "{}.parquet".format(self.FILES["TEST_RAKCHAT_FILE"]))
            self.data["val_df"].to_parquet(fname)
        
        self.logger.info("  done parsing & creating chat data for modelling...")
    
        
    def parse_dialog(self, args):
        if args is None:
            args = self.args
            
        self.logger.info("Parse Raw Chat Data into Consumable Format for Processing:")
        save_dir = os.path.join(args.output_dir, "rakbank_dialog")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        
        file = os.path.join(args.raw_dir, "{}.json".format("chat"))
        with open(file) as f:
            data = json.loads(f.read())
        
        self.logger.info("  putting dialogues into mutliple lists format...")
        parsed_dials = []
        for line in data:
            dialogue = line["dialogue"]
            
            all_text = []
            for text in dialogue:
                text = text["text"]
                all_text.append(text)

            parsed_dials.append({"persona1": [], "persona2": [], "turns": all_text})
            dial_sent_lens = [len(s) for turn in parsed_dials for s in turn["turns"]]

            with open(os.path.join(save_dir, f"parsed_rakbank_dialog.pickle"), "wb") as f:
                pickle.dump(parsed_dials, f)
                
        self.logger.info("  done parsing...")
            
        return parsed_dials, dial_sent_lens


    def split_trn_val(self, df, args):
        if args is None:
            args = self.args
            
        self.logger.info("Create Train & Validation Dataset using Split Ratio at {}:".format(args.train_frac))    
        # concat_dials_lists = sum(dials_list, [])
        # concat_dials_series = pd.Series(concat_dials_lists)
        trn_df, val_df = train_test_split(df, train_size=args.train_frac, shuffle=True)
        
        self.logger.info("  done train test split...")
        
        return trn_df, val_df


    def get_counter_and_lens(self, data:pd.Series, tokenizer):
        flatten = lambda l: [item for sublist in l for item in sublist]
        toks = [tokenizer.tokenize(x) for x in data]
        
        return list(map(len, toks)), Counter(flatten(toks)), Counter(' '.join(data).split())


    def plot_counts(self, counts, top_k:int=30):
        labels, values = zip(*counts.most_common()[:top_k])

        indexes = np.arange(len(labels))
        width = 1
        plt.figure(num=None, figsize=(22, 4), dpi=60, facecolor="w", edgecolor="k")
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        
        return plt.show()


    def plot_hist(self, lens, n_bins:int=50):
        fig, ax = plt.subplots(figsize=(22, 4))
        n, bins, patches = plt.hist(lens, n_bins, facecolor="blue", alpha=0.9)
        
        return plt.show()


def parse_daily_dialog(args):
    raw_dir = os.path.join(Config.FILES["DATA_LOCAL_DIR"], "ParlAI", "data", "dailydialog")
    save_dir = os.path.join(args.output_dir, "daily_dialog")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    files = glob(f"{raw_dir}/*.json")
    
    num_train_dials, num_valid_dials, num_test_dials = 0, 0, 0
    num_train_utters, num_valid_utters, num_test_utters = 0, 0, 0
    min_num_pers, max_num_pers, total_num_pers = 1e+8, 0, 0
    min_num_turns, max_num_turns, total_num_turns = 1e+8, 0, 0
    for file in files:
        print(file)
        prefix = file.split("\\")[-1].split(".")[0]  # "train" or "valid" or "test"
        assert prefix in ["train", "valid", "test"]
        
        dials = [json.loads(line) for line in open(file, 'r')]
        parsed_dials = []
        for dial in tqdm(dials):
            dialogue = dial['dialogue']
            turns = []
            for turn in dialogue:
                text = turn['text']
                turns.append(text)
                
            min_num_pers = min(min_num_pers, 0)
            max_num_pers = max(max_num_pers, 0)
            total_num_pers += 0
            
            min_num_turns = min(min_num_turns, len(turns))
            max_num_turns = max(max_num_turns, len(turns))
            total_num_turns += len(turns)
            
            parsed_dials.append({"persona1": [], "persona2": [], "turns": turns})
        
        num_dials = len(parsed_dials)
        num_utters = 0
        for dial in parsed_dials:
            num_utters += len(dial["turns"])
            
        if prefix == "train":
            num_train_dials = num_dials
            num_train_utters = num_utters
        elif prefix == "valid":
            num_valid_dials = num_dials
            num_valid_utters = num_utters
        elif prefix == "test":
            num_test_dials = num_dials
            num_test_utters = num_utters
        
        with open(os.path.join(save_dir, f"{prefix}.pickle"), 'wb') as f:
            pickle.dump(parsed_dials, f)
            
    data_info ={
        "num_train_dials": num_train_dials,
        "num_valid_dials": num_valid_dials,
        "num_test_dials": num_test_dials,
        "num_train_utters": num_train_utters,
        "num_valid_utters": num_valid_utters,
        "num_test_utters": num_test_utters,
        "max_num_pers": max_num_pers,
        "min_num_pers": min_num_pers,
        "avg_num_pers": round(total_num_pers / (num_train_dials + num_valid_dials + num_test_dials), 2),
        "max_num_turns": max_num_turns,
        "min_num_turns": min_num_turns,
        "avg_num_turns": round(total_num_turns / (num_train_dials + num_valid_dials + num_test_dials), 2),
    }
    
    with open(os.path.join(save_dir, "data_info.json"), 'w') as f: 
        json.dump(data_info, f)
    
    return data_info


def parse_empathetic_dialogues(args):
    comma_symbol = "_comma_"
    
    raw_dir = os.path.join(Config.FILES["DATA_LOCAL_DIR"], "ParlAI", "data", "empatheticdialogues", "empatheticdialogues")
    save_dir = os.path.join(args.output_dir, "empathetic_dialogues")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    files = glob(f"{raw_dir}/*.csv")

    num_train_dials, num_valid_dials, num_test_dials = 0, 0, 0
    num_train_utters, num_valid_utters, num_test_utters = 0, 0, 0
    min_num_pers, max_num_pers, total_num_pers = 1e+8, 0, 0
    min_num_turns, max_num_turns, total_num_turns = 1e+8, 0, 0
    for file in files:
        print(file)
        # prefix = file.split("/")[-1].split(".")[0]  # "train" or "valid" or "test"
        prefix = file.split("\\")[-1].split(".")[0]  # "train" or "valid" or "test"
        assert prefix in ["train", "valid", "test"]
        
        with open(file, "r", encoding="utf8", errors="ignore") as f:
            lines = f.readlines()
            
        cur_conv_id = ""
        dials, turns = [], []
        for l, line in enumerate(tqdm(lines)):
            comps = line.strip().split(",")
            
            if l == 0:
                continue
            
            if cur_conv_id != comps[0]:
                if len(turns) > 0:
                    min_num_pers = min(min_num_pers, 0)
                    max_num_pers = max(max_num_pers, 0)
                    total_num_pers += 0
                    
                    min_num_turns = min(min_num_turns, len(turns))
                    max_num_turns = max(max_num_turns, len(turns))
                    total_num_turns += len(turns)
                    
                    dials.append({"persona1": [], "persona2": [], "turns": turns})
                turns = []
            else:
                assert int(comps[4]) != int(lines[l-1].strip().split(",")[4])
                
            cur_conv_id = comps[0]
            utter = comps[5]

            turns.append(utter.replace(comma_symbol, ","))
            
        if len(turns) > 0:
            min_num_pers = min(min_num_pers, 0)
            max_num_pers = max(max_num_pers, 0)
            total_num_pers += 0
            
            min_num_turns = min(min_num_turns, len(turns))
            max_num_turns = max(max_num_turns, len(turns))
            total_num_turns += len(turns)
            
            dials.append({"persona1": [], "persona2": [], "turns": turns})
            
        num_dials = len(dials)
        num_utters = 0
        for dial in dials:
            num_utters += len(dial["turns"])
            
        if prefix == "train":
            num_train_dials = num_dials
            num_train_utters = num_utters
        elif prefix == "valid":
            num_valid_dials = num_dials
            num_valid_utters = num_utters
        elif prefix == "test":
            num_test_dials = num_dials
            num_test_utters = num_utters
            
        with open(os.path.join(save_dir, f"{prefix}.pickle"), "wb") as f:
            pickle.dump(dials, f)
            
    data_info ={
        "num_train_dials": num_train_dials,
        "num_valid_dials": num_valid_dials,
        "num_test_dials": num_test_dials,
        "num_train_utters": num_train_utters,
        "num_valid_utters": num_valid_utters,
        "num_test_utters": num_test_utters,
        "max_num_pers": max_num_pers,
        "min_num_pers": min_num_pers,
        "avg_num_pers": round(total_num_pers / (num_train_dials + num_valid_dials + num_test_dials), 2),
        "max_num_turns": max_num_turns,
        "min_num_turns": min_num_turns,
        "avg_num_turns": round(total_num_turns / (num_train_dials + num_valid_dials + num_test_dials), 2),
    }
    
    with open(f"{save_dir}/data_info.json", 'w') as f:
        json.dump(data_info, f)
    
    return data_info
    

def parse_persona_chat(args):
    raw_dir = os.path.join(Config.FILES["DATA_LOCAL_DIR"], "ParlAI", "data", "Persona-Chat", "personachat")
    save_dir = os.path.join(args.output_dir, "persona_chat")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    files = glob(f"{raw_dir}/*_self_original.txt")
    
    num_train_dials, num_valid_dials, num_test_dials = 0, 0, 0
    num_train_utters, num_valid_utters, num_test_utters = 0, 0, 0
    min_num_pers, max_num_pers, total_num_pers = 1e+8, 0, 0
    min_num_turns, max_num_turns, total_num_turns = 1e+8, 0, 0
    for file in files:
        print(file)
        # prefix = file.split("/")[-1].split("_")[0]  # "train" or "valid" or "test"
        prefix = file.split("\\")[-1].split(".")[0]  # "train" or "valid" or "test"
        assert prefix in ["train_self_original", "valid_self_original", "test_self_original"]
        
        with open(file, 'r') as f:
            lines = f.readlines()
            
        cur_idx = 0
        dials, turns, pers = [], [], []
        for line in tqdm(lines):
            idx = line.strip().split(" ")[0]

            if cur_idx+1 != int(idx):
                assert int(idx) == 1
                if len(turns) > 0 and len(pers) > 0:
                    min_num_pers = min(min_num_pers, len(pers))
                    max_num_pers = max(max_num_pers, len(pers))
                    total_num_pers += len(pers)

                    min_num_turns = min(min_num_turns, len(turns))
                    max_num_turns = max(max_num_turns, len(turns))
                    total_num_turns += len(turns)
                    
                    dials.append({"persona1": [], "persona2": pers, "turns": turns})
                turns, pers = [], []

            if "\t" in line: # utter
                sliced = line.strip()[len(idx):].strip()
                utters = sliced.split("\t\t")[0]
                turns += utters.split("\t")
            else: # persona
                persona = line.split("your persona:")[-1].strip()
                pers.append(persona)
            cur_idx = int(idx)
            
        if len(turns) > 0 and len(pers) > 0:
            min_num_pers = min(min_num_pers, len(pers))
            max_num_pers = max(max_num_pers, len(pers))
            total_num_pers += len(pers)

            min_num_turns = min(min_num_turns, len(turns))
            max_num_turns = max(max_num_turns, len(turns))
            total_num_turns += len(turns)
            
            dials.append({"persona1": [], "persona2": pers, "turns": turns})
            
        num_dials = len(dials)
        num_utters = 0
        for dial in dials:
            num_utters += len(dial["turns"])
            
        if prefix == "train":
            num_train_dials = num_dials
            num_train_utters = num_utters
        elif prefix == "valid":
            num_valid_dials = num_dials
            num_valid_utters = num_utters
        elif prefix == "test":
            num_test_dials = num_dials
            num_test_utters = num_utters
        
        with open(os.path.join(save_dir, f"{prefix}.pickle"), 'wb') as f:
            pickle.dump(dials, f)
    
    data_info ={
        "num_train_dials": num_train_dials,
        "num_valid_dials": num_valid_dials,
        "num_test_dials": num_test_dials,
        "num_train_utters": num_train_utters,
        "num_valid_utters": num_valid_utters,
        "num_test_utters": num_test_utters,
        "max_num_pers": max_num_pers,
        "min_num_pers": min_num_pers,
        "avg_num_pers": round(total_num_pers / (num_train_dials + num_valid_dials + num_test_dials), 2),
        "max_num_turns": max_num_turns,
        "min_num_turns": min_num_turns,
        "avg_num_turns": round(total_num_turns / (num_train_dials + num_valid_dials + num_test_dials), 2),
    }
    
    with open(os.path.join(save_dir, "data_info.json"), 'w') as f: 
        json.dump(data_info, f)
    
    return data_info


def parse_blended_skill_talk(args):
    raw_dir = os.path.join(Config.FILES["DATA_LOCAL_DIR"], "ParlAI", "data", "blended_skill_talk")
    save_dir = os.path.join(args.output_dir, "blended_skill_talk")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    files = glob(f"{raw_dir}/*.json")
    files = [file for file in files if "_" not in file.split("\\")[-1]]

    num_train_dials, num_valid_dials, num_test_dials = 0, 0, 0
    num_train_utters, num_valid_utters, num_test_utters = 0, 0, 0
    min_num_pers, max_num_pers, total_num_pers = 1e+8, 0, 0
    min_num_turns, max_num_turns, total_num_turns = 1e+8, 0, 0
    for file in files:
        print(file)
        # prefix = file.split("/")[-1].split(".")[0]  # "train" or "valid" or "test"
        prefix = file.split("\\")[-1].split(".")[0]  # "train" or "valid" or "test"
        assert prefix in ["train", "valid", "test"]
        
        with open(file, 'r', encoding="utf8", errors="ignore") as f:
            data = json.load(f)
        
        dials = []
        for dialogue in tqdm(data):
            personas = dialogue["personas"]
            persona1, persona2 = personas[0], personas[1]
            
            dial = {"persona1": persona1, "persona2": persona2}
            turns = [dialogue["free_turker_utterance"], dialogue["guided_turker_utterance"]]
            
            for turn in dialogue["dialog"]:
                turns.append(turn[-1])
                    
            dial["turns"] = turns
            dials.append(dial)
            
            min_num_pers = min(min_num_pers, min(len(persona1), len(persona2)))
            max_num_pers = max(max_num_pers, max(len(persona1), len(persona2)))
            total_num_pers += (len(persona1) + len(persona2))

            min_num_turns = min(min_num_turns, len(turns))
            max_num_turns = max(max_num_turns, len(turns))
            total_num_turns += len(turns)

        num_dials = len(dials)
        num_utters = 0
        for dial in dials:
            num_utters += len(dial["turns"])
            
        if prefix == "train":
            num_train_dials = num_dials
            num_train_utters = num_utters
            print(num_train_dials, num_train_utters)
        elif prefix == "valid":
            num_valid_dials = num_dials
            num_valid_utters = num_utters
            print(num_valid_dials, num_valid_utters)
        elif prefix == "test":
            num_test_dials = num_dials
            num_test_utters = num_utters
            print(num_test_dials, num_test_utters)

        with open(os.path.join(save_dir, f"{prefix}.pickle"), 'wb') as f:
            pickle.dump(dials, f)

    data_info ={
        "num_train_dials": num_train_dials,
        "num_valid_dials": num_valid_dials,
        "num_test_dials": num_test_dials,
        "num_train_utters": num_train_utters,
        "num_valid_utters": num_valid_utters,
        "num_test_utters": num_test_utters,
        "max_num_pers": max_num_pers,
        "min_num_pers": min_num_pers,
        "avg_num_pers": round(total_num_pers / (num_train_dials + num_valid_dials + num_test_dials), 2),
        "max_num_turns": max_num_turns,
        "min_num_turns": min_num_turns,
        "avg_num_turns": round(total_num_turns / (num_train_dials + num_valid_dials + num_test_dials), 2),
    }

    with open(os.path.join(save_dir, "data_info.json"), 'w') as f: 
        json.dump(data_info, f)
    
    return data_info