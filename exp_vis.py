import numpy as np
import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

features = np.arange(2,50,2)[:16]
print(len(features))

mm = 10
fig, ax = plt.subplots(4, 4,
                       figsize=(mm*1.618,mm),
                       sharex=True, sharey=True)

for id, f in enumerate(features):
    # print(id)
    # exit()

    concepts = np.load('streams/all_%i_concepts.npy' % f)
    dbnames = np.load('streams/all_%i_dbnames.npy' % f)
    n_chunks = concepts[-1]
    chunk_size=250

    if n_chunks > 250:
        n_chunks = 250
        concepts = concepts[concepts<n_chunks]

    stream = sl.streams.NPYParser('streams/all_%i.npy' % f,
                        n_chunks=n_chunks, chunk_size=chunk_size)

    clf = MLPClassifier(hidden_layer_sizes=(20))

    scores= []

    while chunk := stream.get_chunk():
        if stream.chunk_id == n_chunks:
            break

        # Get current chunk
        X, y = chunk
        y = y.astype(int)

        # print(X.shape, y.shape)

        # Only train on first chunk
        if stream.chunk_id == 0:
            [clf.partial_fit(X, y, np.unique(y)) for i in range(10)]

        else:
            """
            Test
            """
            # Establish predictions
            y_pred = clf.predict(X)

            # Establish scores
            score = sl.metrics.balanced_accuracy_score(y, y_pred)

            scores.append(score)
            # print(len(scores))
            """
            Train
            """
            [clf.partial_fit(X, y) for i in range(10)]

    id1 = int(id/4)
    id2 = id%4

    # acum = np.linspace(1, n_chunks, len(scores))
    # acscores = np.cumsum(scores) / acum

    sc = gaussian_filter1d(scores, sigma=2)

    aa = ax[id1, id2]

    for cidx, concept in enumerate(concepts):
        aa.text(concept-12, .5, dbnames[cidx],
                ha='center',
                va='center',
                rotation=90,
                color='tomato')


    ax[id1, id2].plot(sc, color='black')
    ax[id1, id2].vlines(concepts, 0, 1, color='tomato', ls=':')
    ax[id1, id2].set_title("%i features" % f)
    ax[id1, id2].grid(color='gray', linestyle=':', linewidth=.3, axis='y')
    ax[id1, id2].set_ylim(0, 1)
    ax[id1, id2].set_xlim(0, 250)
    ax[id1, id2].spines['top'].set_visible(False)
    ax[id1, id2].spines['right'].set_visible(False)

    if id2==0:
        ax[id1, id2].set_ylabel('Accuracy score')
    if id1==2:
        ax[id1, id2].set_xlabel('Processed chunks')


plt.tight_layout()
plt.savefig("figures/exp_vis.eps")
plt.savefig("figures/exp_vis.png")
exit()
