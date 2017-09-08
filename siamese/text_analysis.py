# standard imports
import string
from collections import Counter

# third-party imports
import textacy
from spacy.en import English
from statistics import mean, stdev

def _normalize_counter(counter, c):
    """Divide all the values in a Counter by a constant and remove padding"""
    for key in counter:
        counter[key] = (counter[key] - 1) / c
    return counter

class TextAnalyser:
    def __init__(self, nlp=None):
        if nlp:
            self.nlp = nlp
        else:
            self.nlp = English()
            
        # alphabet for letter ratios
        self.alphabet = string.ascii_lowercase + "!?:;,.'- "
        
        # keys that we care about from textacy.stats
        self.basic_keys = ['n_long_words', 'n_monosyllable_words', 'n_polysyllable_words', 'n_sents', 'n_syllables', 'n_unique_words', 'n_words']
        
        # keys that we care about for textacy readability stats
        self.readability_keys = ['automated_readability_index','coleman_liau_index', 'flesch_kincaid_grade_level',
                                 'flesch_readability_ease', 'gulpease_index', 'gunning_fog_index', 'lix',
                                 'wiener_sachtextformel']
        
        # parts of speech that we care about from spacy (pos_ not tag_)
        self.pos_keys = ['ADJ', 'ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SPACE', 'SYM', 'VERB', 'X']
        self.pos_keys_set = set(self.pos_keys)

    def get_named_features(self, text):
        # TODO: Add bigrams, trigrams?
        processed = self.nlp(text, entity=False, tag=True, parse=True)
        stats = textacy.text_stats.TextStats(processed)
        basic_stats = stats.basic_counts
        readability_stats = stats.readability_stats
        cleaned_text = ''.join(filter(lambda x: x in self.alphabet, text.lower() + self.alphabet))
        
        stats_ratios = {key: (basic_stats[key] / len(text)) for key in self.basic_keys}
        readability_ratios = {key: (readability_stats[key] / len(text)) for key in self.readability_keys}
        stats_ratios.update(readability_ratios)

        # get only the characters we care about 
        # append alphabet so that each character artificially appears once
        char_ratios = Counter(cleaned_text)
        char_ratios = _normalize_counter(char_ratios, len(text))

        # calculate pos ratios
        tags = [word.pos_ for word in processed if word.pos_ in self.pos_keys_set] + self.pos_keys
        pos_ratios = Counter(tags)
        pos_ratios = _normalize_counter(pos_ratios, len(processed)) # normalize by word length

        res = stats_ratios
        res.update(char_ratios)
        res.update(pos_ratios)
        return [(key, res[key]) for key in sorted(res)]
    
    def calculate_mean_and_std(self, extracted_texts):
        """finds unusual patterns by calculating mean and std deviation for a list of 
           extracted features and sorting by z-score"""
        means = []
        stds = []
        sample = extracted_texts[0]  # get one text for feature size and names
        num_features = len(sample)
        # fi = feature index
        for fi in range(num_features):
            u = mean([stat[fi][1] for stat in extracted_texts])
            o = stdev([stat[fi][1] for stat in extracted_texts])
            means.append((sample[fi][0], u))
            stds.append((sample[fi][0], o))
        return means, stds
    
    def calculate_z_scores(self, extracted_text, means, stds):
        """Calculate the zscores for each features of a single text (extractions)"""
        # z = (X - μ) / σ
        zscores = []
        num_features = len(extracted_text)
        for fi in range(num_features):
            feature_name = extracted_text[fi][0]
            try:
                zscore = (extracted_text[fi][1] - means[fi][1]) / stds[fi][1]
            except ZeroDivisionError:
                zscore = 0
            zscores.append((zscore, feature_name))
        return zscores
       
def format_unusual_features(zscores, nfeatures=5):
    zscores = sorted(zscores)
    s = ""
    for j in range(nfeatures):
        fname = zscores[j][1]
        padding = " " * (30 - len(fname))
        s += "{} is low {}(zscore: {})\n".format(fname, padding, zscores[j][0])
    s += "    . . .     \n"
    for j in range(nfeatures):
        fname = zscores[-(j + 1)][1]
        padding = " " * (30 - len(fname))
        s += "{} is high{}(zscore: {})\n".format(fname, padding, zscores[-(j + 1)][0])
    return s

def supports_and_opposes(zscores1, zscores2, sup_threshold=1.5):
    supports = []
    opposes = []
    for i in range(len(zscores1)):
        # both are high or low
        if (zscores1[i][0] > sup_threshold and zscores2[i][0] > sup_threshold) or (
            zscores1[i][0] < -sup_threshold and zscores2[i][0] < -sup_threshold):
            supports.append((zscores1[i], zscores2[i]))
        # one is high and the other is low
        if (zscores1[i][0] > sup_threshold and zscores2[i][0] < -sup_threshold) or (
            zscores1[i][0] < -sup_threshold and zscores2[i][0] > sup_threshold):
            opposes.append((zscores1[i], zscores2[i]))
    return supports, opposes 

def predict(known_texts, unknown_texts):
    ta = TextAnalyser()
    knownstats = [ta.get_named_features(text) for text in known_texts]
    unknownstats = [ta.get_named_features(text) for text in unknown_texts]
    means, stds = ta.calculate_mean_and_std(knownstats + unknownstats)
    knownzs = [ta.calculate_z_scores(ks, means, stds) for ks in knownstats]
    unknownzs = [ta.calculate_z_scores(ks, means, stds) for ks in unknownstats]
    
    preds = []
    for i in range(len(known_texts)):
        z1 = knownzs[i]
        z2 = unknownzs[i]
        s, o = supports_and_opposes(z1, z2, sup_threshold=0.2)
        preds.append(len(s) > len(o))
    return preds
