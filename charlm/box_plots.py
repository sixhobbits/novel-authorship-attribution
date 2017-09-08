# coding: utf-8
from statistics import mean
import pickle
from matplotlib import pyplot as plt
import collections
import os
import csv

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
 

with open("generic_and_fine_scores.pickle", "br") as f:
    scores = pickle.load(f)

truth = load_truth_dict("/data/pan15-authorship-verification-training-dataset-english-2015-04-19/")
    

diff_gen_fine = [scores[score][1] - scores[score][3] for score in scores]

   
same_scores = {score: scores[score] for score in scores if truth[score] == "Y"}
diff_scores = {score: scores[score] for score in scores if truth[score] == "N"}

ss = [scores[score] for score in same_scores]
ds = [scores[score] for score in diff_scores]

m_same_unk_gen_m_fine = mean([score[1] - score[3] for score in ss])
m_diff_unk_gen_m_fine = mean([score[1] - score[3] for score in ds])
diff_unk_gen_m_fine = [score[1] - score[3] for score in ds]
same_unk_gen_m_fine = [score[1] - score[3] for score in ss]

scores = {'E' + score: scores[score] for score in scores}
scores['EN043'] = scores['EEN043']
del scores['EEN043']

same_scores = {score: scores[score] for score in scores if truth[score] == "Y"}
diff_scores = {score: scores[score] for score in scores if truth[score] == "N"}

ss = [scores[score] for score in same_scores]
ds = [scores[score] for score in diff_scores]

m_same_unk_gen_m_fine = mean([score[1] - score[3] for score in ss])
m_diff_unk_gen_m_fine = mean([score[1] - score[3] for score in ds])
diff_unk_gen_m_fine = [score[1] - score[3] for score in ds]
same_unk_gen_m_fine = [score[1] - score[3] for score in ss]

plt.boxplot([same_unk_gen_m_fine, diff_unk_gen_m_fine], title="Difference generic/fine tuned LMs"
)
plt.boxplot([same_unk_gen_m_fine, diff_unk_gen_m_fine])
plt.show()
