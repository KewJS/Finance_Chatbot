import sys
import logging
import numpy as np
from textwrap import TextWrapper
from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import nltk
from nltk.stem.porter import PorterStemmer

import torch
import datasets
import huggingface_hub
import transformers

stemmer = PorterStemmer()
is_colab = "google.colab" in sys.modules
is_kaggle = "kaggle_secrets" in sys.modules
is_gpu_available = torch.cuda.is_available()
    
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = [0, 1, 0, 1, 0, 0, 0]
    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
            
    return bag


def install_mpl_fonts():
    font_dir = ["./orm_fonts/"]
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)
        
        
def set_plot_style():
    install_mpl_fonts()
    set_matplotlib_formats("pdf", "svg")
    plt.style.use("static/plotting.mplstyle")
    logging.getLogger("matplotlib").setLevel(level=logging.ERROR)
    
    
def display_library_version(library):
    print(f"Using {library.__name__} v{library.__version__}")
    
    
def setup_chapter():
    if not is_gpu_available:
        print("No GPU was detected! This notebook can be *very* slow without a GPU 🐢")
        if is_colab:
            print("Go to Runtime > Change runtime type and select a GPU hardware accelerator.")
        if is_kaggle:
            print("Go to Settings > Accelerator and select GPU.")
    
    display_library_version(transformers)
    display_library_version(datasets)
    
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    
    if huggingface_hub.__version__ == "0.0.19":
        huggingface_hub.logging.set_verbosity_error()
    set_plot_style()
    
    
def wrap_print_text(print):
    """
    Adapted from: https://stackoverflow.com/questions/27621655/how-to-overload-print-function-to-expand-its-functionality/27621927
    print = wrap_print_text(print)
    """

    def wrapped_func(text):
        if not isinstance(text, str):
            text = str(text)
        wrapper = TextWrapper(
            width=80,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        return print("\n".join(wrapper.fill(line) for line in text.split("\n")))

    return wrapped_func