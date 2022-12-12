import argparse
import os
import pickle

import numpy as np
from polyaxon.tracking import Run
from sklearn.metrics import adjusted_rand_score, silhouette_score

import metrics.dim_reduction as dim_red
from metrics.dunn import dunn_index
from divae.divae import VAE, DiVAE, AutoGMM

# Polyaxon
experiment = Run()


def path(file):
    if os.path.exists(file) and os.path.isfile(file):
        return file
    raise ValueError(f"File not found or not a file: {file}")


def model(X, max_clusters):
    estimator = DiVAE(auto_gmm=AutoGMM(max_clusters=max_clusters,
                                       random_state=42,
                                       method="youden"),
                      vae=VAE(intermediate_dim=512, latent_dim=2, epochs=250,
                              random_state=42, batch_size=256,
                              verbose=0, shuffle=True),
                      verbose=True, minimal_size=1000, random_state=42).fit(X)
    vae = VAE(intermediate_dim=512, latent_dim=2, epochs=250,
              random_state=42, batch_size=256,
              verbose=0, shuffle=True).fit(X)
    labels = estimator.predict(X)
    embed = vae.transform(X)
    return estimator, embed, labels


def load_data(fname):
    if fname.lower().endswith(".npy"):
        data = np.load(fname)
        return data
    raise ValueError(f'Unsupported file format: {fname}')


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=path, default="")
parser.add_argument('--true_labels', type=path, default="")
parser.add_argument('--max_clusers', type=int, default=20)
args = parser.parse_args()

# data is normalized inside the model
X = load_data(args.data).T
y = load_data(args.true_labels)


# Polyaxon
# https://polyaxon.com/docs/experimentation/tracking/module/#log_data_ref

experiment.log_data_ref('dataset_X', content=X)
experiment.log_data_ref('dataset_y', content=y)

estimator, embed, labels = model(X=X, max_clusters=args.max_clusters)
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

outpath = os.path.join(experiment.get_outputs_path(), 'model.pkl')
with (open(outpath, 'wb')) as outfile:
    pickle.dump(estimator, outfile)

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

experiment.log_model(
    outpath,
    name='DiVAE model',
    framework='sklearn'
)
