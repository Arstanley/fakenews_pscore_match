import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model.logistic import LogisticRegression
from tqdm.auto import tqdm

class propensity_score:
    def __init__(self, model='logistic'):
        self.model = 'logistic'

    def regress_on_words(self, word_index, X):
        """
        word: The word that we are interested in
        text_corpus: input
        """
        labels = []
        tmp_X = X  # Avoid directly changing the variable
        for idx, sentence in enumerate(X):
            if (sentence[word_index] == 1):
                # tmp_X[idx][word_index] = 0
                labels.append(1)
            else:
                labels.append(0)

        # Build the logistic regression model
        log_reg = LogisticRegression()
        log_reg.fit(tmp_X, labels)

        probs = log_reg.predict_proba(tmp_X)[:, -1]

        return probs

    def search_for_closest_control(self, X, idx, word_idx):
        i, j = idx - 1, idx + 1
        while (i >= 0 or j < len(X)):
            if (X[i][word_idx] != 0 and X[j][word_idx] != 0):
                i -= 1
                j += 1
            else:
                if X[i][word_idx] == 1:
                    return X[i], i
                else:
                    return X[j], j

    def calc_chi_square(self, paired_X, word_index):
        # calculate number of treatment-positive and control negative
        TP, CN = 0, 0
        for idx, obj in enumerate(paired_X):
            if obj[word_index] == 1 and obj[-2] == 1:
                TP += 1
            if obj[word_index] == 0 and obj[-2] == 0:
                CN += 1
        print (TP, CN)
        if TP+CN == 0:
            return 0
        else:
            return (TP-CN)**2 / (TP+CN)

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
        X = tokenizer.texts_to_matrix(texts)
        max_len = max([len(arr) for arr in X])
        res = []

        for (word, word_idx) in tqdm(word_dict.items(), total=len(word_dict)): # k: index v: words
            probs = self.regress_on_words(word_idx, X)
            labels, probs = np.array(labels).reshape(864, 1), probs.reshape(864, 1)
            X_with_probas = sorted(np.concatenate((X, labels, probs), axis=1), key=lambda x: -x[-1])
            paired_X = []  # Object that stores the paired instances for Chi-square Calculation
            tmp_X = X_with_probas
            for (idx, treatment) in enumerate(tmp_X):
                if treatment[word_idx] == 1.: # If we could find a treatment element
                    paired_control, ctrl_idx = self.search_for_closest_control(tmp_X, idx, word_idx)
                    paired_X.append(paired_control)
                    paired_X.append(treatment)
                    np.delete(tmp_X, idx, axis=0)
                    np.delete(tmp_X, ctrl_idx, axis=0)
                else:
                    continue
            test_statistics = self.calc_chi_square(paired_X, word_idx) # Calculate the Chi-square statistics for feature selection
            res.append([word, test_statistics])


        self.features = sorted(res, key=lambda x: x[-1])








