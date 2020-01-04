import pandas as pd 
import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model.logistic import LogisticRegression

class propensity_score:
    def __init__(self, model='logistic'):
        self.model = 'logistic'
        ##  Hard-coded for now for the problems in using nltk
        self.stop_words = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 
        'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 
        'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 
        'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 
        'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 
        'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 
        'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 
        'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 
        'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 
        'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

    def regress_on_words(word_index, X):
        """
        word: The word that we are interested in
        text_corpus: input
        """
        labels = []
        for idx, sentence in enumerate(X):
            if (sentence[word_index] == 1):
                X[idx][word_index] = 0
                labels.append(1)
            else:
                labels.append(0)
        
        # Build the logistic regression model
        log_reg = LogisticRegression()
        log_reg.fit(X, labels)

        probs = log_reg.predict_proba(X)[1]

        

        return 

    def fit(self, text_corpus):
        """
        Input: dataframe-like, column[0]: texts, columns[1]: labels
        """
        texts = text_corpus.iloc[:, 0]
        labels = text_corpus.iloc[:, 1]
        
        tokenizer = Tokenizer(num_words=5000, lower=True)  # Parameter num_words will not be used
        tokenizer.fit_on_texts(texts)
        word_dict = tokenizer.word_index
        
        # Transform the data and pad the length for processing
        max_len = max([len(arr) for arr in X])
        X = tokenizer.texts_to_matrix(texts)

        for (k, v) in word_dict: # k: index v: words
            


