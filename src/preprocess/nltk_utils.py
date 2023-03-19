import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

from src.config import Config

stemmer = PorterStemmer()

class NLTK_Utils(Config):
    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)


    def stem(self, word):
        return stemmer.stem(word.lower())


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