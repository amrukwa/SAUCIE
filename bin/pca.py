import argparse
import os
import pickle

import numpy as np
from polyaxon.tracking import Run
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import metrics.dim_reduction as dim_red

# Polyaxon
experiment = Run()


def path(file):
    if os.path.exists(file) and os.path.isfile(file):
        return file
    raise ValueError(f"File not found or not a file: {file}")


def model(X):
    transformer = PCA(2).fit(X)
    results = transformer.transform(X)
    return transformer, results


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

# Polyaxon
# https://polyaxon.com/docs/experimentation/tracking/module/#log_data_ref

experiment.log_data_ref('dataset_X', content=X)
experiment.log_data_ref('dataset_y', content=y)

transformer, results = model(X=X)
silhouette = silhouette_score(results, y)
original_ratios, _ = dim_red.frac_unique_neighbors(X, y, 2)
ratios, _ = dim_red.frac_unique_neighbors(results, y, 2)

sub_label = dim_red.get_optimal_label(X, y)
sub_mat = np.squeeze(X[np.where(y == sub_label), :])
sub_lat = np.squeeze(results[np.where(y == sub_label), :])
var = dim_red.get_variances(sub_mat, sub_lat)

experiment.log_metrics(silhouette_score=silhouette)
experiment.log_metrics(original_mixing_ratio=np.mean(original_ratios))
experiment.log_metrics(reduced_mixing_ratio=np.mean(ratios))
experiment.log_metrics(amb_var=np.mean(var[0]))
experiment.log_metrics(sub_var=np.mean(var[1]))

outpath = os.path.join(experiment.get_outputs_path(), 'model.pkl')
with (open(outpath, 'wb')) as outfile:
    pickle.dump(transformer, outfile)

result_path = os.path.join(experiment.get_outputs_path(), 'pca.csv')
with (open(result_path, 'wb')) as outfile:
    np.savetxt(outfile, results, delimiter=",")

result_path = os.path.join(experiment.get_outputs_path(),
                           'original_ratios.csv')
with (open(result_path, 'wb')) as outfile:
    np.savetxt(outfile, original_ratios, delimiter=",")

result_path = os.path.join(experiment.get_outputs_path(),
                           'reduced_ratios.csv')
with (open(result_path, 'wb')) as outfile:
    np.savetxt(outfile, ratios, delimiter=",")

experiment.log_model(
    outpath,
    name='PCA model',
    framework='sklearn'
)
