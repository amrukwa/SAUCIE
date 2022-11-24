import argparse
import os
import pickle

import numpy as np
from polyaxon.tracking import Run
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from metrics.dunn import dunn_index

# Polyaxon
experiment = Run()


def path(file):
    if os.path.exists(file) and os.path.isfile(file):
        return file
    raise ValueError(f"File not found or not a file: {file}")


def model(X, k_min, k_max):
    estimator = KMeans(k_min).fit(X)
    labels = estimator.predict(X)
    dunn = dunn_index(X, labels, 'euclidean', estimator.cluster_centers_)
    for i in range(k_min+1, k_max+1):
        new_estimator = KMeans(i).fit(X)
        new_labels = new_estimator.predict(X)
        new_dunn = dunn_index(X, new_labels, 'euclidean',
                              new_estimator.cluster_centers_)
        if new_dunn > dunn:
            dunn = new_dunn
            labels = new_labels
            estimator = new_estimator
    return estimator, labels


def load_data(fname):
    if fname.lower().endswith(".npy"):
        data = np.load(fname)
        return data
    raise ValueError(f'Unsupported file format: {fname}')


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=path, default="")
parser.add_argument('--true_labels', type=path, default="")
args = parser.parse_args()

X = load_data(args.data).T
y = load_data(args.true_labels)
min_clusters = args.min_clusters
max_clusters = args.max_clusters

# Polyaxon
# https://polyaxon.com/docs/experimentation/tracking/module/#log_data_ref

experiment.log_data_ref('dataset_X', content=X)
experiment.log_data_ref('dataset_y', content=y)

estimator, labels = model(X, min_clusters, max_clusters)
silhouette_labels = silhouette_score(X, labels)
ari = adjusted_rand_score(y, labels)


experiment.log_metrics(silhouette_score_labels=silhouette_labels)
experiment.log_metrics(n_clusters=np.unique(labels).shape[0])
experiment.log_metrics(ari=ari)

outpath = os.path.join(experiment.get_outputs_path(), 'model.pkl')
with (open(outpath, 'wb')) as outfile:
    pickle.dump(estimator, outfile)

result_path = os.path.join(experiment.get_outputs_path(), 'labels.csv')
with (open(result_path, 'wb')) as outfile:
    np.savetxt(outfile, labels, delimiter=",")


experiment.log_model(
    outpath,
    name='k-means model',
    framework='sklearn'
)
