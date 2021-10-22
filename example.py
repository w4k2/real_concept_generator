import methods
import numpy as np
from tqdm import tqdm

for d in tqdm(range(2,30)):
    real_stream = methods.make_real_stream(d, random_state=1410)

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
