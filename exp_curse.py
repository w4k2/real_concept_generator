from strlearn.streams import NPYParser
from strlearn.ensembles import SEA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score as metric
from sklearn.base import clone
from tqdm import tqdm
from time import time
import numpy as np
import matplotlib.pyplot as plt

ww = 7
fig, ax = plt.subplots(2, 1, figsize=(ww, ww))
#ax = ax.ravel()

# Iterate streams
n_chunks = 1000
sconfigs = [('all_%i' % d, n_chunks, d)
            for d in range(2,26)]


categories = {
    'a': (.5, .6),
    'B': (.6, .7),
    'C': (.7, .8),
    'D': (.8, .9),
    'E': (.9, 1),
}
labels = [
    'all',
    '60-70%',
    '70-80%',
    '80-90%',
    '90-100%'
]
tomato = '#F2704D'
blubato = '#4D70F2'

colors = [
    ('black', '#DDDDDD'),
    ('#F2704D', '#FFDDDD'),
    ('#70F24D', '#DDFFDD'),
    ('#4D70F2', '#DDDDFF'),
    ('#DD55DD', '#FFDDFF'),
]

for cidx, category in enumerate(categories):
    score_to_dim = []
    std_to_dim = []
    stream_lengths = []

    for stream_idx, (sname, n_chunks, d) in enumerate(sconfigs):
        # Generate stream
        stream = NPYParser('streams/%s.npy' % sname,
                            n_chunks=n_chunks, chunk_size=250)

        concepts = np.load('streams/%s_all_%i_concepts.npy' % (category, d))
        dbnames = np.load('streams/%s_all_%i_dbnames.npy' % (category, d))
        scores = np.load('streams/%s_all_%i_scores.npy' % (category, d))
        db = np.load('streams/%s_all_%i.npy' % (category, d))

        stream_lengths.append(db.shape[0])
        print(category, db.shape[0])

        mean_scores = np.mean(scores)
        std_scores = np.std(scores)

        score_to_dim.append([d, mean_scores])
        std_to_dim.append([d, std_scores])


    score_to_dim = np.array(score_to_dim).T
    std_to_dim = np.array(std_to_dim).T
    print(score_to_dim.shape)

    if cidx != 0:
        ax[0].fill_between(score_to_dim[0],
                        score_to_dim[1]-std_to_dim[1],
                        score_to_dim[1]+std_to_dim[1],
                        color=colors[cidx][1])
    ax[0].plot(score_to_dim[0], score_to_dim[1],
               c=colors[cidx][0], label=labels[cidx],
               ls='-' if cidx!=0 else ":")

    ax[1].plot(score_to_dim[0], stream_lengths, label=labels[cidx],
               c=colors[cidx][0],
               ls='-' if cidx!=0 else ":")

for i in range(2):
    ax[i].grid(ls=':')
    ax[i].legend(frameon=False, ncol=5, loc=4)
    ax[i].set_xlim(2,25)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].set_xlabel('Problem dimensionality')

ax[0].set_ylabel('Average accuracy')
ax[1].set_ylabel('Number of samples')
ax[0].set_ylim(.5,1)
ax[1].set_ylim(0, 300000)

plt.tight_layout()

plt.savefig('figures/exp_curse.png')
plt.savefig('figures/exp_curse.eps')

exit()
