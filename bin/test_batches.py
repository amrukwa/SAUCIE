import argparse
import os

import numpy as np
from polyaxon.tracking import Run
from polyaxon.tracking.contrib.keras import PolyaxonCallback
from tensorflow.keras.callbacks import EarlyStopping

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

result_path = os.path.join(experiment.get_outputs_path(), 'normalized.npy')
with (open(result_path, 'wb')) as outfile:
    np.save(outfile, X, delimiter=",")

# Polyaxon
# https://polyaxon.com/docs/experimentation/tracking/module/#log_data_ref

experiment.log_data_ref('dataset_X', content=X)
experiment.log_data_ref('dataset_y', content=y)
experiment.log_data_ref('batches', content=batches)

# Add model and batch corrected data
models, results = model(X=X, batches=batches)

transformer, estimator = models
embed, labels, cleaned_data = results

result_path = os.path.join(experiment.get_outputs_path(), 'cleaned_data.npy')
with (open(result_path, 'wb')) as outfile:
    np.save(outfile, cleaned_data, delimiter=",")

result_path = os.path.join(experiment.get_outputs_path(), 'embed.npy')
with (open(result_path, 'wb')) as outfile:
    np.save(outfile, embed, delimiter=",")

result_path = os.path.join(experiment.get_outputs_path(), 'labels.npy')
with (open(result_path, 'wb')) as outfile:
    np.save(outfile, labels, delimiter=",")
