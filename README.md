# Real concept generator

This module contains generator of the data streams from real concepts which can be found in the file `metods.py`. For that, generator uses benchmark datasets from `datasets`.

The method takes arguments described below:
- `base_clf` – base classification method used in the member problem projection audit;
- `base_metric` – base metric used in the projection audit;
- `base_directory` – path to the folder with datasets describing member problems;
- `tag_filter` – list of tags for filtering member problems;
- `random_state` – random state used in generating projections and in SMOTE algorithm;
- `min_samples` – the minimum number of samples that the member problems must contain to be included in generated data stream;
- `n_projections` – limit on the number of random projections generated;
- `metric_treshold`– a tuple of the metric threshold range;
- `stream_requirements`– generated data stream length requirements; sequentially: the size of the chunk and the minimum number of chunks;
- `sampling_strategy` – sampling strategy for the SMOTE algorithm.

## Examples of usage

Example of usage of the method can be found in file `example.py`. Files `exp_vis.py`, `exp_det.py`, `exp_curse.py` and `exp_concept.py` contain examples of the visualization with multiple analysis - eg. features and drift detection previews. Result of them can be seen in directory `figures`.
