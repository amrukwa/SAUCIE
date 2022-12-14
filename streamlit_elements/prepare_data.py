import numpy as np


def filter_data(data, frac=1.0):
    data = data.dropna(axis=0)
    data = data.loc[:, ~(data == 0).all(axis=0)]
    data.drop(columns=data.columns[data.dtypes == object], inplace=True)
    if frac != 1.0:
        x = np.var(data, axis=0)
        thr = (-1)*np.sort(-x)[int(x.shape[0]*frac)]
        is_in_filtered = x > thr
        data = data[is_in_filtered.index[is_in_filtered]]
    columns = data.columns.tolist()
    data = data.to_numpy()
    data = data.astype(float)
    return data, columns


def extract_metalabel(data, label):
    metalabel = data[[label]].to_numpy()
    metalabel = metalabel.flatten()
    data = data.drop(columns=label)
    return data, metalabel


def normalize_data(data, normalize=True):
    if normalize:
        data = data/1000
        data = data - np.min(data, axis=0)
        data = data/np.max(data, axis=0)
        data = np.arcsinh(data)
    return data
