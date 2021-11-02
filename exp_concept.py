import numpy as np
import matplotlib.pyplot as plt

f=2

concepts = np.load('streams/all_%i_concepts.npy' % f)
dbnames = np.load('streams/all_%i_dbnames.npy' % f)

n_chunks = concepts[-1]
chunk_size=250

if n_chunks > 250:
    n_chunks = 250
    concepts = concepts[concepts<n_chunks]

db = np.load('streams/all_%i.npy' % f)

concepts_diff = concepts[1]-concepts[0]
print(concepts_diff)

concepts_mids = concepts - .5*concepts_diff
concepts_mids = concepts_mids[:10]
print(concepts_mids)

mm=10
fig, ax = plt.subplots(4, 2, figsize=(mm*.8,mm*1.618))

vvv=2
fig, ax = plt.subplots(5, vvv,
                       figsize=(mm/1.618,mm))


for id, c in enumerate(concepts_mids.astype(int)):
    id1=int(id/2)
    id2=int(id%2)

    print(id, id1, id2)
    ch_start = c*250
    ch_end = (c+1)*250

    X = db[ch_start:ch_end,:-1]
    y = db[ch_start:ch_end,-1]

    ax[id1, id2].scatter(X[y==0,0],
                         X[y==0,1],
                         c = 'tomato',
                         marker='o',
                         alpha=1)

    ax[id1, id2].scatter(X[y==1,0],
                         X[y==1,1],
                         c = 'black',
                         marker='o',
                         alpha=1)

    ax[id1, id2].set_title(dbnames[id])
    ax[id1, id2].set_xlabel("chunk %i" % c)

    xl = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 5)
    ax[id1, id2].set_xticks(xl)
    ax[id1, id2].set_xticklabels(['%.1f' % f for f in xl], fontsize=8)


    yl = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 5)
    ax[id1, id2].set_yticks(yl)
    ax[id1, id2].set_yticklabels(['%.1f' % f for f in yl], fontsize=8)

    ax[id1, id2].grid(color='#CCC', linestyle=':', linewidth=1)

    ax[id1, id2].spines['top'].set_visible(False)
    ax[id1, id2].spines['right'].set_visible(False)

    if id1 == 4:
        ax[id1, id2].set_xlabel('feature 0')

    if id2 == 0:
        ax[id1, id2].set_ylabel('feature 1')

plt.tight_layout()
plt.savefig("figures/exp_concept.eps")
plt.savefig("figures/exp_concept.png")
