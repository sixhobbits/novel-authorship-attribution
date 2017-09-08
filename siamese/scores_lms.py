# coding: utf-8

with open(scores.txt) as f:
    s = f.read()
    
with open('scores.txt') as f:
    s = f.read()
    
    
lines = s.split("\n")
lines[0]
scores = {}
for line in lines:
    score, tup = line.split(":")
    scores[score] = eval(tup)
    
scores
len(scores)
scores['EN001']
diff_gen_fine = [scores[score][1] - scores[score][3] for score in scores]
diff_gen_fine
sum([1 for score in diff_gen_fine if score > 0])
with open("/data/pan15-authorship-verification-training-dataset-english-2015-04-19/truth.txt") as f:
    lines = f.read().split("\n")
    
lines[0]
truth = {line.split()[0]: line.split()[1] for line in lines}
truth
len(lines)
lines[-1]
del lines[-1]
len(lines)
truth = {line.split()[0]: line.split()[1] for line in lines}
truth
same_scores = {score: scores[score] for score in scores if truth[score] == "Y"}
len(same_scores)
diff_scores = {score: scores[score] for score in scores if truth[score] == "N"}
len(diff_scores)
ss = [scores[score] for score in scores]
ss
ss = [scores[score] for score in same_scores]
len(ss)
ss
ds = [scores[score] for score in diff_scores]
ds
len(ds)
from statistics import mean
mean([score[0] for score in same])
mean([score[0] for score in ss])
mean([score[0] for score in ds])
mean([score[1] for score in ds])
mean([score[1] for score in ss])
mean([score[2] for score in ss])
mean([score[2] for score in ds])
mean([score[3] for score in ds])
mean([score[3] for score in ss])
same_unk_gen_m_fine = mean([score[1] - score[3] for score in ss])
diff_unk_gen_m_fine = mean([score[1] - score[3] for score in ds])
same_unk_gen_m_fine
diff_unk_gen_m_fine
diff_unk_gen_m_fine = [score[1] - score[3] for score in ds]
same_unk_gen_m_fine = [score[1] - score[3] for score in ss]
