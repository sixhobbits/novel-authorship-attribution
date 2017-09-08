# coding: utf-8

from pphelper import vectorize_text
from panhelper import get_all_authors
from keras.utils import np_utils
from models import get_basic_model
import numpy as np
from config import SEQ_LENGTH
from pphelper import ALPHABET
import sys

def formatX(X, seq_length=SEQ_LENGTH):
    """Normalize and reshape X; one-hot encode Y"""
    X = np.reshape(X, (len(X), seq_length, 1))
    X = X / float(len(ALPHABET))
    return X

train_path = "/data/pan15-authorship-verification-training-dataset-english-2015-04-19/"
authors = get_all_authors(train_path)


for i in range(len(authors)):
    authors[i].id = i

Xs = [vectorize_text(author.unknown) for author in authors]
Xs = [formatX(x) for x in Xs]

model = get_basic_model()
model.load_weights("author_id.h5")
preds = model.predict(Xs[4])
combined = []
for i in range(100):
    combined.append(sum([pred[i] for pred in preds]))
    
combined[0]
from matplotlib import pyplot as plt
get_ipython().magic('save id_analysis.py 0-6')
plt.bar(range(100), combined)
plt.show()
preds
plt.show()
plt.bar(range(100), combined)
plt.show()
def graph_one(Xs, model, ind):
    preds = model.predict(Xs[ind])
    combined = [sum([pred[i] for pred in preds]) for i in range(100)]
    plt.bar(range(100), combined)
    plt.show()
    
