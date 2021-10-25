import numpy as np
from methods import make_real_stream
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

for d in range(2,61):
    print(d)
    # Configure stream
    config = {
        'd': d, 'random_state':1410, 'n_projections':500,
        'metric_treshold': (.55, 1.1), 'base_clf': GaussianNB(),
        'base_metric': accuracy_score, 'base_directory': 'datasets/',
        'tag_filter': ['binary'], 'min_samples': 200,
        'stream_requirements':(250, 100),
        'sampling_strategy':{0: 3000, 1: 3000}
    }

    # Prepare stream
    real_stream = make_real_stream(**config)

    # Store stream
    if real_stream:
        db, concepts, dbnames, scores = real_stream
        np.save('streams/all_%i_concepts' % d, concepts)
        np.save('streams/all_%i_dbnames' % d, dbnames)
        np.save('streams/all_%i_scores' % d, scores)
        np.save('streams/all_%i' % d, db)

        print('CONCEPTS', concepts)
        print('DBNAMES', dbnames)
        print('SCORES', scores)
