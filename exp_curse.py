from strlearn.streams import NPYParser
from strlearn.ensembles import SEA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score as metric
from sklearn.base import clone
from tqdm import tqdm
from time import time
import numpy as np
import matplotlib.pyplot as plt

ww = 5
fig, ax = plt.subplots(1, 1, figsize=(ww*1.618, ww))
#ax = ax.ravel()

# Iterate streams
n_chunks = 1000
sconfigs = [('all_%i' % d, n_chunks, d)
            for d in range(2,26)]

score_to_dim = []
std_to_dim = []

for stream_idx, (sname, n_chunks, d) in enumerate(sconfigs):
    # Generate stream
    stream = NPYParser('streams/%s.npy' % sname,
                        n_chunks=n_chunks, chunk_size=250)

    concepts = np.load('streams/all_%i_concepts.npy' % d)
    dbnames = np.load('streams/all_%i_dbnames.npy' % d)
    scores = np.load('streams/all_%i_scores.npy' % d)

    mean_scores = np.mean(scores)
    std_scores = np.std(scores)

    score_to_dim.append([d, mean_scores])
    std_to_dim.append([d, std_scores])


score_to_dim = np.array(score_to_dim).T
std_to_dim = np.array(std_to_dim).T
print(score_to_dim)

ax.fill_between(score_to_dim[0],
                score_to_dim[1]-std_to_dim[1],
                score_to_dim[1]+std_to_dim[1],
                color='#CCCCCC')
ax.plot(score_to_dim[0], score_to_dim[1],
        c='black')
ax.grid(ls=':')
ax.set_xlim(2,25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Problem dimensionality')
ax.set_ylabel('Full-model accuracy')
ax.set_ylim(.5,1)

plt.tight_layout()

plt.savefig('figures/exp_curse.png')
plt.savefig('figures/exp_curse.eps')

exit()
