# standard imports
import sys
import os

# local imports
from models import get_basic_model
from pphelper import text2vec

def train(text, model, save_file):
    X, y = text2vec(text)
    model.fit(X, y, nb_epoch=50, batch_size=64, validation_split=0.2)
    print("saving to: {}".format(save_file))
    model.save_weights(save_file)
    return model

def train_fine(new_text, generic_weights, save_file):
    with open(new_text) as f:
        s = f.read()
    model = get_basic_model(freeze_feature_layers=True)
    model.load_weights(generic_weights)
    fine_model = train(s, model, save_file)
    return fine_model

def evaluate(textfile, model):
    with open(textfile) as f:
        s = f.read()
    X, y = text2vec(s)
    return model.evaluate(X, y)


def evaluate_4way(known_file, unknown_file, generic_model, fine_weights):
    
    fine_model = get_basic_model()
    fine_model.load_weights(fine_weights)

    known_fine = evaluate(known_file, fine_model)
    unknown_fine = evaluate(unknown_file, fine_model)
    known_gen = evaluate(known_file, generic_model)
    unknown_gen = evaluate(unknown_file, generic_model)

    return known_gen, unknown_gen, known_fine, unknown_fine


def main():
    directory = sys.argv[1]
    generic_w = sys.argv[2]
    generic_model = get_basic_model()
    generic_model.load_weights(generic_w)

    authors = [x for x in os.listdir(directory) if x.startswith("EN")]
    scores = {}
    for author in authors:
        print("working on {}".format(author))
        known = os.path.join(directory, author, "known01.txt")
        unknown = os.path.join(directory, author, "unknown.txt")
        known_weights = os.path.join(directory, author, "knownmodel.h5")
        kg, ug, kf, uf = evaluate_4way(known, unknown, generic_model, known_weights)
        scores[author] = (kg, ug, kf, uf)
        print("author: {} kg: {}, ug: {}, kf: {}, uf: {}".format(author, kg, ug, kf, uf)) 
    print(scores)
    with open("scores.txt", "w") as f:
        for author in scores:
            f.write("{}: {}".format(author, scores[author]))



if __name__ == "__main__":
    main()




