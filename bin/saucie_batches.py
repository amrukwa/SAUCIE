import argparse
import os

import numpy as np
from polyaxon.tracking import Run
from polyaxon.tracking.contrib.keras import PolyaxonCallback
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tensorflow.keras.callbacks import EarlyStopping

import metrics.dim_reduction as dim_red
from metrics.dunn import dunn_index
from saucie.saucie import SAUCIE_batches, SAUCIE_labels

# Polyaxon
experiment = Run()


def path(file):
    if os.path.exists(file) and os.path.isfile(file):
        return file
    raise ValueError(f"File not found or not a file: {file}")


def model(X, batches):
    transformer = SAUCIE_batches(lr=1e-4,
                                 batch_size=256,
                                 epochs=200,
                                 random_state=123,
                                 callback=[EarlyStopping(monitor='loss',
                                           patience=50,
                                           restore_best_weights=True),
                                           PolyaxonCallback(log_model=False)],
                                 verbose=True).fit(X, batches)
    cleaned_data = transformer.transform(X, batches)
    estimator = SAUCIE_labels(lr=1e-4, shuffle=True,
                              batch_size=256, verbose=0,
                              random_state=123,
                              callback=[PolyaxonCallback(log_model=False)]
                              ).fit(X)
    embed = estimator.transform(X)
    labels = estimator.predict(X)
    return [transformer, estimator], [embed, labels, cleaned_data]


def load_data(fname):
    if fname.lower().endswith(".npy"):
        data = np.load(fname)
        return data
    raise ValueError(f'Unsupported file format: {fname}')


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=path, default="")
parser.add_argument('--true_labels', type=path, default="")
parser.add_argument('--batches', type=path, default="")
args = parser.parse_args()

X = load_data(args.data).T
y = load_data(args.true_labels)
batches = load_data(args.batches)

# normalize data for SAUCIE
X = X - np.min(X, axis=0)
X = X/np.max(X, axis=0)
X = np.arcsinh(X)

# Polyaxon
# https://polyaxon.com/docs/experimentation/tracking/module/#log_data_ref

experiment.log_data_ref('dataset_X', content=X)
experiment.log_data_ref('dataset_y', content=y)
experiment.log_data_ref('batches', content=batches)

# Add model and batch corrected data
models, results = model(X=X, batches=batches)

transformer, estimator = models
embed, labels, cleaned_data = results

silhouette_embed = silhouette_score(embed, y)
silhouette_labels = silhouette_score(X, labels)
ari = adjusted_rand_score(y, labels)
dunn = dunn_index(X, labels, 'euclidean', None)
original_ratios, _ = dim_red.frac_unique_neighbors(X, y, 2)
ratios, _ = dim_red.frac_unique_neighbors(embed, y, 2)

sub_label = dim_red.get_optimal_label(X, y)
sub_mat = np.squeeze(X[np.where(y == sub_label), :])
sub_lat = np.squeeze(embed[np.where(y == sub_label), :])
var = dim_red.get_variances(sub_mat, sub_lat)

experiment.log_metrics(silhouette_score_embed=silhouette_embed)
experiment.log_metrics(silhouette_score_labels=silhouette_labels)
experiment.log_metrics(n_clusters=np.unique(labels).shape[0])
experiment.log_metrics(dunn_idx=dunn)
experiment.log_metrics(ari=ari)
experiment.log_metrics(original_mixing_ratio=np.mean(original_ratios))
experiment.log_metrics(reduced_mixing_ratio=np.mean(ratios))
experiment.log_metrics(amb_var=np.mean(var[0]))
experiment.log_metrics(sub_var=np.mean(var[1]))

result_path = os.path.join(experiment.get_outputs_path(), 'cleaned_data.csv')
with (open(result_path, 'wb')) as outfile:
    np.savetxt(outfile, cleaned_data, delimiter=",")

result_path = os.path.join(experiment.get_outputs_path(), 'embed.csv')
with (open(result_path, 'wb')) as outfile:
    np.savetxt(outfile, embed, delimiter=",")

result_path = os.path.join(experiment.get_outputs_path(), 'labels.csv')
with (open(result_path, 'wb')) as outfile:
    np.savetxt(outfile, labels, delimiter=",")

result_path = os.path.join(experiment.get_outputs_path(),
                           'original_ratios.csv')
with (open(result_path, 'wb')) as outfile:
    np.savetxt(outfile, original_ratios, delimiter=",")

result_path = os.path.join(experiment.get_outputs_path(),
                           'reduced_ratios.csv')
with (open(result_path, 'wb')) as outfile:
    np.savetxt(outfile, ratios, delimiter=",")
