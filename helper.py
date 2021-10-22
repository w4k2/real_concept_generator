import os
import re
import numpy as np

def dir2files(path, extention="csv"):
    return [
        path + x
        for x in os.listdir(path)
        if re.match("^([a-zA-Z0-9-_])+\.%s$" % extention, x)
    ]

def tags4Xy(X, y):
    tags = []
    numberOfFeatures = X.shape[1]
    numberOfSamples = len(y)
    numberOfClasses = len(np.unique(y))
    if numberOfClasses == 2:
        tags.append("binary")
    else:
        tags.append("multi-class")
    if numberOfFeatures >= 8:
        tags.append("multi-feature")

    # Calculate ratio
    ratio = [0.0] * numberOfClasses
    for y_ in y:
        ratio[y_] += 1
    ratio = [int(round(i / min(ratio))) for i in ratio]
    if max(ratio) > 4:
        tags.append("imbalanced")

    return tags

def csv2Xy(path):
    ds = np.genfromtxt(path, delimiter=",")
    X = ds[:, :-1]
    y = ds[:, -1].astype(int)
    dbname = path.split("/")[-1].split(".")[0]
    tags = tags4Xy(X, y)
    return X, y, dbname, tags

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def datasets_of_tags(tag_filter, directory='datasets/'):
    files = dir2files(directory)
    datasets = []
    for file in files:
        X, y, dbname, tags = csv2Xy(file)
        intersecting_tags = intersection(tags, tag_filter)
        if len(intersecting_tags):
            datasets.append((X, y, dbname))

    return datasets
