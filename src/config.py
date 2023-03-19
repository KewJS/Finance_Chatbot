import os
import inspect
import fnmatch
from collections import OrderedDict

from nltk.corpus import stopwords
from krovetzstemmer import Stemmer
from nltk.stem import WordNetLemmatizer

base_path, current_dir = os.path.split(os.path.dirname(inspect.getfile(inspect.currentframe())))


class Config(object):
    QDEBUG = True
    
    FILES = dict(
        DATA_LOCAL_DIR      = os.path.join(base_path, "data"),
        RAW_DATA_DIR        = os.path.join(base_path, "data", "raw"),
        PREPROCESS_DIR      = os.path.join(base_path, "data", "preprocess"),
        MODEL_DATA_DIR      = os.path.join(base_path, "data", "models"),
        CORNELL_MOVIE_DIR   = os.path.join(base_path, "data", "raw", "cornell movie-dialogs corpus"),
        
        RAW_INTENTS_FILE    = "intents",
        RAW_CORNELL_CONV    = "movie_conversations",
        RAW_CORNELL_LINES   = "movie_lines",
        
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