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
concepts_mids = concepts_mids[:8]
print(concepts_mids)

mm=10
fig, ax = plt.subplots(4, 2, figsize=(mm,mm*1.618))

for id, c in enumerate(concepts_mids.astype(int)):
    id1=int(id/2)
    id2=int(id%2)

    print(id, id1, id2)
    ch_start = c*250
    ch_end = (c+1)*250
    ax[id1, id2].scatter(db[ch_start:ch_end,0], db[ch_start:ch_end,1], c = db[ch_start:ch_end,2], cmap='copper')
    ax[id1, id2].set_title(dbnames[id])
    ax[id1, id2].set_xlabel("chunk %i" % c)
    ax[id1, id2].grid(color='gray', linestyle=':', linewidth=.3)

    ax[id1, id2].spines['top'].set_visible(False)
    ax[id1, id2].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("figures/exp_concept.eps")
plt.savefig("figures/exp_concept.png")
