import sys

import numpy as np
from keras.models import load_model

from pphelper import text2vec
from pphelper import text2sequence
from pphelper import ALPHABET
from pphelper import i2c

from models import get_basic_model


def generate(model, seed, chars_to_gen=1000):
    predictions = []
    for step in range(chars_to_gen):
        x = np.reshape(seed, (1, len(seed), 1))
        x = x / float(len(ALPHABET))
        prediction = model.predict(x, verbose=0)
        prediction = np.argmax(prediction)
        predictions.append(prediction)
        seed.append(prediction)
        seed = seed[1:]
    return ''.join([i2c(c) for c in predictions])


def main():
    weightsfile = sys.argv[1]
    textfile = sys.argv[2]

    model = get_basic_model(freeze_feature_layers=True)
    model.load_weights(weightsfile)
    
    with open(textfile) as f:
        s = f.read()
    X, y = text2vec(s)

    print(model.evaluate(X, y))
    # X, y = text2sequence(s)
    # seed = X[0]
    # print(type(seed), len(seed))
    # print(seed)
    # print("-------")

    # result = generate(model, seed, 3000)
    # print(result)


if __name__ == "__main__":
    main()


