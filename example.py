import methods
import numpy as np
from tqdm import tqdm

for d in tqdm(range(1,30)):
    real_stream = methods.make_real_stream(d)

    if real_stream is not None:
        db, concepts = real_stream
        np.save('streams/all_%i_drifts' % d, concepts)
        np.save('streams/all_%i' % d, db)
