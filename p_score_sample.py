from models.propensity_score import propensity_score, info_gain
import os, pickle
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Stop Words obtained from NLTK

stop_words = {'a', 'ourselves', 'hers', 'between', 'yourself',
              'but', 'again', 'there', 'about', 'once', 'during',
              'out', 'very', 'having', 'with', 'they', 'own', 'an',
              'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
              'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or',
              'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until',
              'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don',
              'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down',
              'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
              'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before',
              'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
              'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so',
              'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has',
              'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i',
              'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against',
              'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 's', 'i', 't', 'To'
             'u', '3', '2', '6', '5', '60', 'u'}

# Read data
politifact_real = pd.read_csv('./FakeNewsNet-master/dataset/politifact_real.csv')
politifact_fake = pd.read_csv('./FakeNewsNet-master/dataset/politifact_fake.csv')
gossipcop_real = pd.read_csv('./FakeNewsNet-master/dataset/gossipcop_real.csv')
gossipcop_fake = pd.read_csv('./FakeNewsNet-master/dataset/gossipcop_fake.csv')
len(politifact_real), len(politifact_fake), len(gossipcop_real), len(gossipcop_fake)

politifact_real = politifact_real.sample(432)
gossipcop_real = gossipcop_real.sample(432)
gossipcop_fake = gossipcop_fake.sample(432)

politifact_real['label'] = 1
politifact_fake['label'] = 0
gossipcop_real['label'] = 1
gossipcop_fake['label'] = 0

politifact = pd.concat((politifact_fake, politifact_real), axis=0)
gossipcop = pd.concat((gossipcop_real, gossipcop_fake), axis=0)

len(politifact), len(gossipcop)

politifact = politifact[['title', 'label']]
gossipcop = gossipcop[['title', 'label']]

# Get rid of the stop words
def clean_stopwords(sentences):
    """
    input: array of sentences
    """
    word_list = re.findall(r'\w+', sentences)
    ans = ''
    for word in word_list:
        if word.lower() in stop_words:
            continue
        ans += word + ' '
    return ans

politifact['title'] = politifact['title'].map(lambda x: clean_stopwords(x))
gossipcop['title'] = gossipcop['title'].map(lambda x: clean_stopwords(x))

try:
    f = open("gossipcop_pscore.txt", 'rb')
    # Do something with the file
    p_score_gossipcop = pickle.load(f)
except IOError:
    print("File not accessible")
    p_score_gossipcop = propensity_score()
    p_score_gossipcop.fit(gossipcop)
    f = open("gossipcop_pscore.txt", 'wb+')
    pickle.dump(p_score_gossipcop, f)
finally:
    f.close()

