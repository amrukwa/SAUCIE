import argparse
import os

import numpy as np
import pandas as pd
from polyaxon.tracking import Run

# Polyaxon
experiment = Run()


def path(file):
    if os.path.exists(file) and os.path.isfile(file):
        return file
    raise ValueError(f"File not found or not a file: {file}")


def filter_data(data):
    data = data.dropna(axis=1)
    x = np.var(data, axis=1)
    thr = (-1)*np.sort(-x)[int(x.shape[0]*0.2)]
    experiment.log_metrics(thr=thr)
    is_in_filtered = x > thr
    filtered = data[is_in_filtered]

    x = np.log(x+np.full(x.shape, 1))
    result_path = os.path.join(experiment.get_outputs_path(),
                               'variances.csv')
    with (open(result_path, 'wb')) as outfile:
        np.savetxt(outfile, x, delimiter=",")

    filtered_df = pd.DataFrame(filtered)

    filtered = filtered/1000
    result_path = os.path.join(experiment.get_outputs_path(),
                               'filtered_data.npy')
    with (open(result_path, 'wb')) as outfile:
        np.save(outfile, filtered)

    return filtered_df


def save_metadata(metadata, col, name):
    if col == "unbatched_data":
        count = 1
        values = np.zeros(metadata.shape[0])
    else:
        values = np.array(pd.factorize(metadata[col])[0])
        unique_val, _ = np.unique(metadata[col], return_counts=True)
        count = unique_val.shape[0]
    result_path = os.path.join(experiment.get_outputs_path(),
                               name)
    with (open(result_path, 'wb')) as outfile:
        np.save(outfile, values)
    return count


def get_metadata_df(metadata, col, idx):
    vals = metadata[col].to_numpy().reshape(1, metadata.shape[0])
    df_vals = pd.DataFrame(vals, columns=metadata[idx],
                           index=["cell_type"])
    return df_vals


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=path, default="")
parser.add_argument('--metadata', type=path, default="")
parser.add_argument('--label_col', type=str, default="CellType")
parser.add_argument('--name_col', type=str, default="NAME")
parser.add_argument('--batch_col', type=str, default="unbatched_data")

args = parser.parse_args()

# filter data and get statistics
data = pd.read_csv(args.data, index_col=0)
experiment.log_metrics(n_genes=data.shape[0])
data = filter_data(data)
experiment.log_metrics(n_genes=data.shape[0])

# get metalabels
meta = pd.read_csv(args.metadata, index_col=0)

# ground truths
n_clusters = save_metadata(meta, args.label_col, 'true_labels.npy')
experiment.log_metrics(n_clusters=n_clusters)

# batches
n_batches = save_metadata(meta, args.batch_col, 'batches.npy')
experiment.log_metrics(n_batches=n_batches)

# get overall csv file
new_labels = get_metadata_df(meta, args.label_col, args.name_col)


new_df = pd.concat([data, new_labels])
if args.batch_col != "unbatched_data":
    batch_df = get_metadata_df(meta, args.batch_col, args.name_col)
    new_df = pd.concat([new_df, batch_df])

new_df = new_df.T
result_path = os.path.join(experiment.get_outputs_path(),
                           'labeled_data.csv')
with (open(result_path, 'wb')) as outfile:
    new_df.to_csv(outfile)
