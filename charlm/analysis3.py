# coding: utf-8

coding: utf-8

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
graph
graph_one
graph_one(Xs, models, 6)
graph_one(Xs, model, 6)
graph_one(Xs, model, 7)
graph_one(Xs, model, 8)
graph_one(Xs, model, 8)
graph_one(Xs, model, 9)
graph_one(Xs, model, 11)
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
import collections 
import csv
from ../../glad/glad-main import load_truth_dict
def load_truth_dict(path):
    """
    Load the truth values for a data-set from the TXT file
    :param path: The path to the directory containing the truth.txt file
    :return: A dictionary with the problem names as keys and the true class labels as values
    """

    truth_dict = collections.defaultdict(str)
    with open(os.path.join(path, 'truth.txt'), 'r', encoding='utf-8-sig') as truth_file:
        truth = csv.reader(truth_file, delimiter=' ')
        for problem in truth:
            truth_dict[problem[0]] = problem[1]
            # log.debug(problem[0], problem[1])
        return truth_dict
    
truth = load_truth_dict("/data/pan15-authorship-verification-training-dataset-english-2015-04-19/")
import os
truth = load_truth_dict("/data/pan15-authorship-verification-training-dataset-english-2015-04-19/")
truth
same = set([key for key in truth if truth[key] == "Y"]
)
same
same = [x[-3:] for x in same]
same
same = [int(x) for x in same]
same
same = [x-1] for x in same]
same = [x-1 for x in same]
same
len(Xs)
len(Xs[same])
type(Xs)
same_xs = [Xs[i] for i in same]
len(same_xs)
