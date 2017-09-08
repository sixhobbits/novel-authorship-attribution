#!/usr/bin/env python

import numpy as np
np.random.seed(1337)  # for reproducibility and comparability, don't change!
import argparse

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils, generic_utils
from gender_nb import parse

from vectorizers import TextStats, Word2Vec

class KerasNN(object):

    def __init__(self, n_units, n_epochs, batch_size, name):

        # Parameters
        self.batch_size          = batch_size
        self.n_epochs            = n_epochs
        self.hidden_units_1      = n_units
        self.name                = name

       #  Run the neural net
        self.load_data()
        self.build_model()
        self.train_model()
        self.test_model()

    def load_data(self):
        print("Loading data...")
        trainX, trainY, testX, testY = parse()

        ts = FeatureUnion([
            ('stats', TextStats()),
            ('tfidf', TfidfVectorizer()),
            ('vector', Word2Vec())
        ])

        self.X_train = ts.fit_transform(trainX)
        self.X_test = ts.fit_transform(testX)

        self.Y_train = [0 if x == "F" else 1 for x in trainY]
        self.Y_test = [0 if x == "F" else 1 for x in testY]

        self.nb_features = self.X_train.shape[1]
        self.nb_classes = 2
        self.Y_train = np_utils.to_categorical(self.Y_train, self.nb_classes)
        self.Y_test = np_utils.to_categorical(self.Y_test, self.nb_classes)

    def build_model(self):
        print("Building model...")
        self.model = Sequential()

        # Comment this...
        self.model.add(Dense(input_dim=self.nb_features, output_dim=self.hidden_units_1))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.50))

        # Comment this...
        self.model.add(Dense(input_dim=self.hidden_units_1, output_dim=self.hidden_units_1))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.50))

        # Comment this...
        self.model.add(Dense(input_dim=self.hidden_units_1, output_dim=self.nb_classes))
        self.model.add(Activation('softmax'))

        # Comment this...
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


    def train_model(self):
        print("Training model...")
        n_instances = len(self.X_train)
        dev_split   = int(n_instances * 0.8)

        X_train, X_val, Y_train, Y_val = train_test_split(self.X_train, self.Y_train, test_size=0.2)

        history = self.model.fit(X_train, Y_train, nb_epoch=self.n_epochs, batch_size=self.batch_size, shuffle='batch', 
                                 verbose=1, validation_data=(X_val, Y_val), )

    def test_model(self):
        outputs = self.model.predict(self.X_test, batch_size=self.batch_size)
        pred_classes = np.argmax(outputs, axis=1)

        np.save(self.name, pred_classes)

parser = argparse.ArgumentParser(description='KerasNN parameters')
parser.add_argument('--units', metavar='xx', type=int, default=500, help='units')
parser.add_argument('--epochs', metavar='xx', type=int, default=20, help='epochs')
parser.add_argument('--bsize', metavar='xx', type=int, default=50, help='batch size')
parser.add_argument('--name', type=str, default='tratz_test_output', help='output file name')
args = parser.parse_args()

if __name__ == '__main__':
    KerasNN(n_units=args.units,
            n_epochs=args.epochs,
            batch_size=args.bsize,
            name=args.name)
