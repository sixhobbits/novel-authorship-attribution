# Helper functions for preprocessing raw text before feeding it into a Neural Net
import string

import numpy as np
import sys
import os
tmp = sys.stderr
null = open(os.devnull, "w")
sys.stderr = null
from keras.utils import np_utils
null.close()
sys.stderr = tmp

from config import SEQ_LENGTH

# Map printable characters to ints and vice-versa
ALPHABET = string.printable

char2int = dict((c, i) for i, c in enumerate(ALPHABET))
int2char = dict((i, c) for i, c in enumerate(ALPHABET))


def c2i(char):
    return char2int[char]

def i2c(num):
    return int2char[num]


def vectorize_text(text, seq_length=SEQ_LENGTH):
    """Convert a text into integers"""
    X = []
    text = ''.join(list(filter(lambda x: x in ALPHABET, text)))
    nchars = len(text)
    for i in range(0, nchars - seq_length, 1):
        seq_in = text[i:i + seq_length]
        X.append([c2i(char) for char in seq_in])
    return X

def text2sequence(text, seq_length=SEQ_LENGTH):
    X = []
    y = []
    text = ''.join(list(filter(lambda x: x in ALPHABET, text)))
    nchars = len(text)
    for i in range(0, nchars - seq_length, 1):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        X.append([c2i(char) for char in seq_in])
        y.append(c2i(seq_out))
    return X, y

def one_hot_encode(y):
    """Consistently one-hot-encodes the labels based on alphabet"""

    # Add all characters to get correct one-hot-encoding
    y += [c2i(c) for c in ALPHABET]
    y = np_utils.to_categorical(y)

    # Remove the bits we added
    y = y[:-len(ALPHABET)]
    return y


def format_x(X, seq_length=SEQ_LENGTH):
    X = np.reshape(X, (len(X), seq_length, 1))
    X = X / float(len(ALPHABET))
    return X


def formatXY(X, y, seq_length=SEQ_LENGTH):
    """Normalize and reshape X; one-hot encode Y"""
    X = np.reshape(X, (len(X), seq_length, 1))
    X = X / float(len(ALPHABET))
    y = one_hot_encode(y) 
    return X, y


def text2vec(text, seq_length=SEQ_LENGTH):
    X, y = text2sequence(text, seq_length=seq_length)
    X, y = formatXY(X, y, seq_length=seq_length)
    return X, y


def vectorize_texts_authors(texts, author_labels):
    """IN: texts - An array of texts
       IN: author_labels - an array of integer author IDs"""
    X = [vectorize_text(text) for text in texts]
    y = np_utils.to_categorical(author_labels)
    return X, y


if __name__ == "__main__":
    print("test")
