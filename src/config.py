import os
import inspect
import fnmatch
from datetime import datetime
from collections import OrderedDict

from nltk.corpus import stopwords
from krovetzstemmer import Stemmer
from nltk.stem import WordNetLemmatizer

base_path, current_dir = os.path.split(os.path.dirname(inspect.getfile(inspect.currentframe())))


class Args():
    def __init__(self):
        self.raw_dir = "data/raw"
        self.train_frac = 0.9
        self.output_dir = "data/output"
        self.model_type = "gpt2"
        self.model_name_or_path = "microsoft/DialoGPT-small"
        self.config_name = "microsoft/DialoGPT-small"
        self.tokenizer_name = "microsoft/DialoGPT-small"
        self.cache_dir = "data/cached"
        self.block_size = 64 # 128
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = False
        self.per_gpu_train_batch_size = 2
        self.per_gpu_eval_batch_size = 2
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 3
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 1000
        self.save_steps = 3500
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = "O1"


class Config(object):
    QDEBUG = True
    
    
    APPS = dict(
        MODEL_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    
    
    FILES = dict(
        DATA_LOCAL_DIR      = os.path.join(base_path, "data"),
        RAW_DATA_DIR        = os.path.join(base_path, "data", "raw"),
        CORNELL_MOVIE_DIR   = os.path.join(base_path, "data", "raw", "cornell movie-dialogs corpus"),
        PREPROCESS_DIR      = os.path.join(base_path, "data", "preprocess"),
        MODEL_DATA_DIR      = os.path.join(base_path, "data", "models"),
        CACHED_DATA_DIR     = os.path.join(base_path, "data", "cached"),
        
        RAW_INTENTS_FILE    = "intents",
        RAW_CORNELL_CONV    = "movie_conversations",
        RAW_CORNELL_LINES   = "movie_lines",
        RAW_BANK_FAQ_FILE   = "BankFAQs",
        
        PROCESS_RAKCHAT_FILE= "process_rakchat",
        TRAIN_RAKCHAT_FILE  = "train_rakchat",
        TEST_RAKCHAT_FILE   = "test_rakchat",
        TRAIN_INTENTS_FILE  = "X_train",
        TEST_INTENTS_FILE   = "X_test",
        ALL_WORDS_LIST      = "all_words_list",
        TAGS_LIST           = "tags_list",
        XY_LIST             = "xy_list",
        
    )

    
    ANALYSIS_CONFIG = dict(
        READ_PREPROCESS     = True,
        NLTK_STOPWORDS      = stopwords.words("english"),
        KROVERTZ_STEMMER    = Stemmer(),
        LEMMATIZER          = WordNetLemmatizer(),
        IGNORE_CHARACTERS   = ["?", "!", ".", ",", ":"],
    )
    
    
    MODELLING_CONFIG = dict(
        # # dataloader
        BATCH_SIZE      = 8,
        SHUFFLE         = True,
        NUM_WORKERS     = 0,
        
        HIDDEN_SIZE     = 8,
        LEARNING_RATE   = 0.001,
        NUM_EPOCHS      = 1000,
        
        BOT_NAME        = "Rak-Voice",
    )
    
    TKINTER_CONFIG = dict(
        BG_GRAY     = "#ABB2B9",
        BG_COLOR    = "#17202A",
        TEXT_COLOR  = "#EAECEE",
        
        FONT        = "Helvetica 14",
        FONT_BOLD   = "Helvetica 13 bold",
    )
    
    
    VARS = OrderedDict(
        REVIEWS = [
            dict(var="rating",          dtypes=float,   predictive=False),
            dict(var="page_number",     dtypes=int,     predictive=True),
            dict(var="book_author",     dtypes=str,     predictive=True),
            dict(var="reviews",         dtypes=str,     predictive=True),
            dict(var="title",           dtypes=str,     predictive=True),
            dict(var="reviews_length",  dtypes=int,     predictive=True),
        ],
        
    )