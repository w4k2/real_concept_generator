import helper
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

def make_real_stream(d=20,
                     base_clf=GaussianNB(),
                     base_metric=accuracy_score,
                     base_directory='datasets/',
                     tag_filter=['binary'],
                     random_state=None,
                     min_samples=200,
                     n_projections=10,
                     metric_treshold=(.55,1),
                     stream_requirements=(250, 100),
                     sampling_strategy={
                         0: 3000,
                         1: 3000
                     }
                     ):
    # Establish random state
    np.random.seed(random_state)

    # Initialize concept containers
    X_s, y_s = [], []

    # Initialize descriptive containers
    dbnames, scores, counts = [], [], []

    # Prepare kf object
    kf = KFold(n_splits=5)

    # Gather all datasets
    datasets = helper.datasets_of_tags(tag_filter,
                                       base_directory)

    # Iterate datasets
    for i, dataset in enumerate(datasets):
        # Extract dataset
        X, y, dbname = dataset

        # Ignore small datasets
        if X.shape[0] < min_samples:
            continue

        # Prepare search for projection
        best = -2.
        bidx = 0
        selected_projection = None

        # Search for projection
        for p_idx in range(n_projections):
            # Randomize projection
            projection = np.random.normal(size=(X.shape[1], d))

            # Project
            X_ = X @ projection

            # Gather CV score
            score = np.mean([base_metric(y[test],clone(base_clf).fit(X_[train], y[train]).predict(X_[test]))
                             for train, test in kf.split(X_)])

            # Store best overall
            if (score > metric_treshold[0]) * (score < metric_treshold[1]):
                selected_projection = projection
                best = score
                bidx = p_idx
                break

        # Verify if dataset meets the requirements
        if best > metric_treshold[0]:
            # Project
            X_ = X @ selected_projection

            # Oversample
            sm = SMOTE(random_state=random_state,
                       sampling_strategy=sampling_strategy)
            X_, y = sm.fit_resample(X_, y)

            # Permute samples
            p = np.random.permutation(len(y))

            # Append dataset
            X_s.append(X_[p])
            y_s.append(y[p])

            # Append parameters
            scores.append(np.copy(best))
            dbnames.append(dbname)
            counts.append(X_.shape[0])

    # Concatenate datasets
    X = np.concatenate(X_s, axis=0)
    y = np.concatenate(y_s, axis=0)
    db = np.concatenate((X, y[:,np.newaxis]), axis=1)

    # Verify if stream meets the requirements
    if len(y)/stream_requirements[1] >= stream_requirements[0]:
        concepts = np.rint(np.cumsum(counts)/250)
        return db, concepts, np.array(dbnames), np.array(scores)
    else:
        return None
