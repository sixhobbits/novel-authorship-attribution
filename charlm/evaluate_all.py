# standard imports
import sys
import os

# local imports
from models import get_basic_model
from pphelper import text2vec

def train(text, model, save_file):
    X, y = text2vec(text)
    model.fit(X, y, nb_epoch=100, batch_size=64, validation_split=0.2)
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

def train_and_evaluate(known, unknown, generic_weights, save_file):
    fine_model = train_fine(known, generic_weights, save_file)

    with open(unknown) as f:
        s = f.read()
    X, y = text2vec(s)
    evaluation = fine_model.evaluate(X, y)

    with open(known) as f:
        s = f.read()
    X, y = text2vec(s)
    base_eval = fine_model.evaluate(X, y)
    return evaluation, base_eval

def main():
    if len(sys.argv) != 4:
        print("usage: python3 evaluate_all.py <fine-text> <save-fine> <generic-weights>")
        return
    textfile = sys.argv[1]
    savefile = sys.argv[2]
    genericf = sys.argv[3]
    train_fine(textfile, genericf, savefile)
    print("done")
    return





def main2():
    directory = sys.argv[1]
    generic_w = sys.argv[2]
    authors = [x for x in os.listdir(directory) if x.startswith("EN")]
    scores = {}
    for author in authors:
        print("working on {}".format(author))
        known = os.path.join(directory, author, "known01.txt")
        unknown = os.path.join(directory, author, "unknown.txt")
        save_file = os.path.join(directory, author, "knownmodel.h5")
        score, baseline = train_and_evaluate(known, unknown, generic_w, save_file)
        scores[author] = (score, baseline)
        print("score: {}, baseline: {}".format(score, baseline))
    print(scores)
    with open("scores.txt", "w") as f:
        for author in scores:
            f.write("{}: {}, {}\n".format(author, scores[author[0]], scores[author[1]]))



if __name__ == "__main__":
    main()




