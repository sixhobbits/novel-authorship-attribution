from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential

from pphelper import ALPHABET
from config import SEQ_LENGTH



def get_basic_model(seq_length=SEQ_LENGTH, features=1, freeze_feature_layers=False):

    feature_layers = [LSTM(256, input_shape=(seq_length, features), return_sequences=True), 
                      LSTM(256)]

    classification_layers = [Dense(len(ALPHABET), activation='softmax')]
    if freeze_feature_layers:
        for layer in feature_layers:
            layer.trainable = False

    model = Sequential(feature_layers + classification_layers)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
