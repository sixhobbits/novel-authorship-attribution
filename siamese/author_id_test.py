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

train_path = sys.argv[1]
authors = get_all_authors(train_path)


for i in range(len(authors)):
    authors[i].id = i
    
Xs = [vectorize_text(author.unknown) for author in authors]
Xs = [formatX(x) for x in Xs]

model = get_basic_model()
model.load_weights("author_id.h5")
for X in Xs:
    pred = model.predict(X)
    print(np.argmax(pred))
