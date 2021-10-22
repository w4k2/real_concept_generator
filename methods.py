import helper
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

def make_real_stream(d=2,
                     base_clf=GaussianNB(),
                     base_metric=accuracy_score):

    X_s = []
    y_s = []
    dbnames = []
    n_s = []
    scores = []
    counts = []

    datasets = helper.datasets_of_tags(['binary'])

    for i, dataset in enumerate(datasets):
        X, y, dbname = dataset
        if X.shape[1] >= d:
            if X.shape[0] < 200:
                continue

            best = -2.
            bidx = 0
            columns = None

            for z in range(100):
                selcons = np.random.choice(list(range(X.shape[1])), size=d, replace=False)

                n = len(y)
                X_ = X[:,selcons]

                clf = clone(base_clf).fit(X_, y)
                y_pred = clf.predict(X_)
                score = base_metric(y, y_pred)

                if score > best:
                    columns = selcons
                    best = score
                    bidx = z

            if best > .75:
                scores.append(np.copy(best))

                X_ = X[:,columns]

                sm = SMOTE(random_state=42,
                           sampling_strategy={
                               0: 3000,
                               1: 3000
                           })
                X_, y = sm.fit_resample(X_, y)

                p = np.random.permutation(len(y))

                X_s.append(X_[p])
                y_s.append(y[p])
                dbnames.append(dbname)
                n_s.append(X.shape[1])
                counts.append(X_.shape[0])

    X = np.concatenate(X_s, axis=0)
    y = np.concatenate(y_s, axis=0)

    db = np.concatenate((X, y[:,np.newaxis]), axis=1)

    if len(y)/250 >= 100:
        print("# D=%i" % d, len(dbnames), db.shape, np.unique(y, return_counts=True), len(y)/250, '%.3f:%.3f:%.3f' % (np.min(scores), np.max(scores), np.mean(scores)))

        concepts = np.rint(np.cumsum(counts)/250)

        return db, concepts


    else:
        return None
