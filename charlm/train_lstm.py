# standard imports
import sys


# local imports
from models import get_basic_model
from pphelper import text2vec
from keras.callbacks import EarlyStopping

def train(text, model, save_file):
    callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    X, y = text2vec(text)
    model.fit(X, y, nb_epoch=50, batch_size=64, validation_split=0.2, callbacks=[callback])
    model.save_weights(save_file)

def train_generic(corpus_file, save_file="model_weights_generic.h5"):
    with open(corpus_file) as f:
        s = f.read()
    model = get_basic_model()
    train(s, model, save_file)

def train_fine(person_file, weightsfile, save_file="model_weights_person.h5"):
    with open(person_file) as f:
        s = f.read()
    model = get_basic_model(freeze_feature_layers=True)
    model.load_weights(weightsfile)
    train(s, model, save_file)

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 train_lstm.py <text-file> <save-weights-file>")
        return
    train_generic(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()




