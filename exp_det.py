import numpy as np
import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from _det import DDM

tomato = '#F2704D'
blubato = '#4D70F2'

mm = 5
fig, ax = plt.subplots(1, 1,
                       figsize=(mm*1.618,mm),
                       sharex=True, sharey=True)

f = 8

concepts = np.load('streams/all_%i_concepts.npy' % f)
dbnames = np.load('streams/all_%i_dbnames.npy' % f)
n_chunks = concepts[-1]
chunk_size=250

if n_chunks > 250:
    n_chunks = 250
    concepts = concepts[concepts<n_chunks]

stream = sl.streams.NPYParser('streams/all_%i.npy' % f,
                    n_chunks=n_chunks, chunk_size=chunk_size)

clf = MLPClassifier(hidden_layer_sizes=(20), random_state=765)
detector = DDM()

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
        detector.feed(X,y,y_pred)

drf_idxs = np.argwhere(np.array(detector.drift)==2)
print(detector.drift, drf_idxs)

sc = gaussian_filter1d(scores, sigma=2)

ax.plot(sc, color='black')
ax.vlines(drf_idxs, 0, 1, color=tomato, ls='-', linewidth = 4)

ax.vlines(concepts, 0, 1, color=blubato, ls=':')
ax.set_title("Drifts detected with DDM")
ax.grid(color='gray', linestyle=':', linewidth=.3, axis='y')
ax.set_ylim(0.4, 1)
ax.set_xlim(0, 250)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Processed chunks')
ax.set_ylabel('Accuracy score')
ax.set_xticks(concepts)

plt.tight_layout()
plt.savefig("figures/exp_det.eps")
plt.savefig("figures/exp_det.png")
