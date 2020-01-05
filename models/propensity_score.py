import pandas as pd 
import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model.logistic import LogisticRegression

class propensity_score:
    def __init__(self, model='logistic'):
        self.model = 'logistic'

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
        return probs

    def search_for_closest_control(X, idx, word_idx):
        i, j = idx - 1, idx + 1
        while (i >= 0 or j < X.shape[0]):
            if (X[i][word_idx] != 0 and X[j][word_idx] != 0):
                i -= 1
                j += 1
            else:
                if X[i][word_idx] == 1:
                    return X[i], i
                else:
                    return X[j], j

    def calc_chi_square(paired_X):
        

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

        for (word_idx, word) in enumerate(word_dict): # k: index v: words
            probs = regress_on_words(word_idx, X)
            X_with_probas = sorted(np.concatenate((X, labels, probs), axis=1), key=lambda x: x[-1])
            paired_X = np.array([])  # Object that stores the paired instances for Chi-square Calculation
            for (idx, treatment) in enumerate(X_with_probas):
                if treatment[word_idx] == 1: # Then we found a treatment element
                    paired_control, ctrl_idx = search_for_closest_control(X_with_probas, idx, word_idx)
                    np.append(paired_X, obj)
                    np.append(paired_X, paired_elt)
                    np.delete(X_with_probas, idx, axis=0)
                    np.delete(X_with_probas, ctrl_idx)
                else:
                    continue
            test_statistics = calc_chi_square(paired_X) # Calculate the Chi-square statistics for feature selection

                


            


