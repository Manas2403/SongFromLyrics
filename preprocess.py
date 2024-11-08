import os
import nltk
import json
import re
import string
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer

class Preprocessing:
    def __init__(self, Pipeline=['']):
        self.Pipeline = Pipeline

        # Tokenizer setup
        self.tokenizer = 'No_tokenize'
        if "nltk_word_tokenizer" in self.Pipeline:
            self.tokenizer = 'nltk'
        elif "word_space_tokenize" in self.Pipeline:
            self.tokenizer = 'split'

        # Initialize Porter Stemmer if in pipeline
        if "PorterStemmer" in self.Pipeline:
            self.stemmer = PorterStemmer()

        # Load contraction dictionary and set stop words
        self.CONTRACTION_MAP = self._load('dictionary')
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.stop_words.remove("not")

    def _load(self, path):
        # Load dictionary file from 'dictionary' folder
        if 'contraction_word_dictionary.txt' not in os.listdir(path):
            print("Could not find 'contraction_word_dictionary.txt' file.")
            return {}
        else:
            with open(os.path.join(path, 'contraction_word_dictionary.txt')) as f:
                return json.loads(f.read())

    def text_lowercase(self, text):
        # Convert to lowercase
        return text.lower()

    def convert_unicode(self, text):
        return text.encode('ascii', 'ignore').decode()

    def delete_tag(self, text):
        # Remove tags like '[Verse 1]', '[Chorus]', '[Intro]' in lyrics
        return re.sub(r'\[(.*?)\]', '', text)

    def remove_whitespace(self, text):
        # Remove extra whitespace
        return " ".join(text.split())

    def remove_stopwords(self, text):
        # Remove stop words
        word_tokens = nltk.tokenize.word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in self.stop_words]
        return " ".join(filtered_text)

    def remove_punctuation(self, text):
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def expand_contraction(self, text):
        # Expand contractions
        return ' '.join([self.CONTRACTION_MAP.get(item, item) for item in text.split()])

    def stem(self, text):
        # Apply stemming with PorterStemmer
        return ". ".join([" ".join([self.stemmer.stem(word) for word in sent.split(' ')]) for sent in text.split('. ')])

    def Preprocess(self, str):
        # Apply the preprocessing steps
        str = self.text_lowercase(str)
        str = self.convert_unicode(str)
        str = self.delete_tag(str)
        str = str.replace('\n\n', '')
        str = str.replace('\n', '. ')
        str = self.remove_whitespace(str)
        str = self.expand_contraction(str)

        if 'PorterStemmer' in self.Pipeline:
            str = self.stem(str)

        str = self.remove_punctuation(str)

        if 'remove_stopword' in self.Pipeline:
            str = self.remove_stopwords(str)
        
        return str
