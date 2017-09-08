# Some helper functions that deal with data in the PAN format
import os
from logger import log

def get_single_author(path):
    """Get list of known texts + single unknown text from the specific path"""
    files = os.listdir(path)
    knowns = []
    unknowns = []
    for fname in files:
        with open(os.path.join(path,fname)) as f:
            s = f.read()
        if fname.startswith("known") and fname.endswith(".txt"):
            knowns.append(s)
        elif fname == "unknown.txt":
            unknowns.append(s)
        else:
            log.warning("Filename {} seems to be neither known or unknown. Skipping".format(fname))
            knowns.append(s)
    return knowns, unknowns


class Author:
    def __init__(self, name, knowntext, unknowntext):
        self.name = name
        self.known = knowntext
        self.unknown = unknowntext


def get_all_authors(path):
    """Gets all examples from single language directory"""
    dirs = os.listdir(path)
    authors = []
    for d in dirs:
        full_dir = os.path.join(path, d)
        if not os.path.isdir(full_dir):  # skip files
            continue
        knowns, unknowns = get_single_author(full_dir)
        a = Author(d, '\n'.join(knowns), '\n'.join(unknowns))
        authors.append(a)
    return authors






        
