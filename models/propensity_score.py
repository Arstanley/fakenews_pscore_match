import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model.logistic import LogisticRegression
from tqdm.auto import tqdm

class info_gain:
    def calc_entropy(self, df):
        labels = df.iloc[:, -1]
        prob_true = np.count_nonzero(labels) / len(labels)
        prob_false = 1 - prob_true
        return self._entropy([prob_true, prob_false])

    def _entropy(self, probs):
        """
        Parameters:
        probs: Array-like
        """
        res = 0
        if (np.sum(probs) != 1):
            print("Probabilities have to sum to 1")
            return False
        for prob in probs:
            res += -np.log(prob) * prob
        return res

    def fit(self, text_corpus):
        texts = text_corpus.iloc[:, 0]
        labels = np.array(text_corpus.iloc[:, 1]).reshape(len(text_corpus), 1)

        tokenizer = Tokenizer(num_words=3200, lower=True)  # Parameter num_words will not be used
        tokenizer.fit_on_texts(texts)
        word_dict = tokenizer.word_index

        # Transform the data and pad the length for processing
        X = tokenizer.texts_to_matrix(texts)
        X = pd.DataFrame(np.concatenate((X, labels), axis = 1))

        res = []

        original_entropy = self.calc_entropy(X)

        for (word, word_idx) in tqdm(word_dict.items(), total=len(word_dict)): # k: index v: words
            # Calculate Entropy After The Split
            splitted_set1 = X[X[word_idx] == 1]
            splitted_set2 = X[X[word_idx] == 1]

            after_entropy = self.calc_entropy(splitted_set1) + self.calc_entropy(splitted_set2)

            res.append([word, after_entropy])

        self.features = sorted(res, key=lambda x: x[-1])

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
            if (X[i][word_idx] == 1 and X[j][word_idx] == 1):
                i -= 1
                j += 1
            else:
                if X[i][word_idx] == 0:
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

        texts = np.array(texts).reshape(864, 1)

        for (word, word_idx) in tqdm(word_dict.items(), total=len(word_dict)): # k: index v: words
            probs = self.regress_on_words(word_idx, X)
            labels, probs = np.array(labels).reshape(864, 1), probs.reshape(864, 1)
            X_with_probas = sorted(np.concatenate((X, labels, probs), axis=1), key=lambda x: -x[-1])
            texts_with_probas = sorted(np.concatenate((texts, probs), axis=1), key=lambda x: -x[-1])
            paired_X = []  # Object that stores the paired instances for Chi-square Calculation
            paired_X_idx = []
            tmp_X = X_with_probas
            for (idx, treatment) in enumerate(tmp_X):
                if treatment[word_idx] == 1.: # If we could find a treatment element
                    paired_control, ctrl_idx = self.search_for_closest_control(tmp_X, idx, word_idx)
                    paired_X.append(paired_control)
                    paired_X.append(treatment)
                    paired_X_idx.append((idx, 0))
                    paired_X_idx.append((ctrl_idx, 1))
                else:
                    continue
            test_statistics = self.calc_chi_square(paired_X, word_idx) # Calculate the Chi-square statistics for feature selection
            res.append([word, test_statistics])


        self.features = sorted(res, key=lambda x: x[-1])








