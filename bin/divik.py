import argparse
import os
import pickle

import numpy as np
from divik.cluster import DiviK
from polyaxon.tracking import Run
from sklearn.metrics import adjusted_rand_score, silhouette_score

from metrics.dunn import dunn_index

# Polyaxon
experiment = Run()


def path(file):
    if os.path.exists(file) and os.path.isfile(file):
        return file
    raise ValueError(f"File not found or not a file: {file}")


def model(X, distance):
    estimator = DiviK(distance=distance).fit(X)
    labels = estimator.labels_
    return estimator, labels


def load_data(fname):
    if fname.lower().endswith(".npy"):
        data = np.load(fname)
        return data
    raise ValueError(f'Unsupported file format: {fname}')


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=path, default="")
parser.add_argument('--true_labels', type=path, default="")
parser.add_argument('--distance', type=str, default='euclidean')
args = parser.parse_args()

# data is normalized inside the model
X = load_data(args.data).T
y = load_data(args.true_labels)


# Polyaxon
# https://polyaxon.com/docs/experimentation/tracking/module/#log_data_ref

experiment.log_data_ref('dataset_X', content=X)
experiment.log_data_ref('dataset_y', content=y)

estimator, labels = model(X=X, distance=args.distance)
silhouette_labels = silhouette_score(X, labels)
ari = adjusted_rand_score(y, labels)
dunn = dunn_index(X, labels, 'euclidean', None)

experiment.log_metrics(silhouette_score_labels=silhouette_labels)
experiment.log_metrics(n_clusters=np.unique(labels).shape[0])
experiment.log_metrics(dunn_idx=dunn)
experiment.log_metrics(ari=ari)

outpath = os.path.join(experiment.get_outputs_path(), 'model.pkl')
with (open(outpath, 'wb')) as outfile:
    pickle.dump(estimator, outfile)

result_path = os.path.join(experiment.get_outputs_path(), 'labels.csv')
with (open(result_path, 'wb')) as outfile:
    np.savetxt(outfile, labels, delimiter=",")

experiment.log_model(
    outpath,
    name='DiviK model',
    framework='sklearn'
)
