import numpy as np
import pandas as pd

# from saucie.saucie import SAUCIE_batches, SAUCIE_labels
from streamlit_elements.prepare_data import (extract_metalabel, filter_data,
                                             normalize_data)


def test_saucie_gets_filtered_data():
    binary = np.load("test/test_data/filtered_data.npy").T
    df = pd.read_csv("test/test_data/labeled_data.csv", index_col=0)
    data, _ = extract_metalabel(df, "run")
    data, _ = extract_metalabel(data, "cell_type")
    data, _ = filter_data(data, frac=1.0)
    data = data/1000
    np.testing.assert_array_equal(data, binary)


def test_saucie_gets_normalized_data():
    binary = np.load("test/test_data/normalized_data.npy")
    df = pd.read_csv("test/test_data/labeled_data.csv", index_col=0)
    data, _ = extract_metalabel(df, "run")
    data, _ = extract_metalabel(data, "cell_type")
    data, _ = filter_data(data, frac=1.0)
    data = normalize_data(data, normalize=True)
    np.testing.assert_array_equal(data, binary)


def test_saucie_gets_batches():
    binary = np.load("test/test_data/batches.npy")
    df = pd.read_csv("test/test_data/labeled_data.csv", index_col=0)
    _, batches = extract_metalabel(df, "run")
    batches = np.unique(batches, return_inverse=True)[1]
    np.testing.assert_array_equal(batches, binary)


# def test_saucie_returns_debatched_data():
#     assert 0 == 0


# def test_saucie_returns_labels():
#     assert 0 == 0


# def test_saucie_returns_embeds():
#     assert 0 == 0
