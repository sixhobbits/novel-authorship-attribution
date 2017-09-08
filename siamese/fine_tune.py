# standard imports
import sys


# local imports
from models import get_basic_model
from pphelper import text2vec

def train(text, model, save_file):
    X, y = text2vec(text)
    model.fit(X, y, nb_epoch=50, batch_size=64, validation_split=0.33)
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
    generic_weights = sys.argv[1]
    new_text = sys.argv[2]
    output_weights = sys.argv[3]
    train_fine(new_text, generic_weights, output_weights)

if __name__ == "__main__":
    main()




