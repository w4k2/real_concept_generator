import methods
import numpy as np
from tqdm import tqdm

for d in tqdm(range(2,50,2)):
    real_stream = methods.make_real_stream(d, random_state=None,
                                           n_projections=2,
                                           metric_treshold=.55,
                                           base_clf=GaussianNB(),
                                           base_metric=accuracy_score,
                                           base_directory='datasets/',
                                           tag_filter=['binary'],
                                           min_samples=200,
                                           stream_requirements=(250, 100),
                                           sampling_strategy={
                                               0: 3000,
                                               1: 3000
                                           })

    if real_stream is not None:
        db, concepts, dbnames, scores = real_stream
        np.save('streams/all_%i_concepts' % d, concepts)
        np.save('streams/all_%i_dbnames' % d, dbnames)
        np.save('streams/all_%i_scores' % d, scores)
        np.save('streams/all_%i' % d, db)

        print('CONCEPTS', concepts)
        print('DBNAMES', dbnames)
        print('SCORES', scores)
        print(concepts.shape,
              dbnames.shape,
              scores.shape)
