import io

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_blobs

from saucie.saucie import SAUCIE_batches, SAUCIE_labels


def data_saucie():
    """
    prepare exemplary test data
    """
    data = make_blobs(n_samples=10000, n_features=20,
                      centers=2, random_state=42)[0]
    data = data - np.min(data)
    return data


def data_batches():
    """
    prepare exemplary test data for batches
    """
    rng = np.random.default_rng(42)
    data = rng.integers(0, 3, 10000)
    return data


def test_SAUCIE_compresses_data():
    """
    test if SAUCIE reduces the data to 2 dimensions
    and does not lose the samples
    """
    data = data_saucie()
    saucie = SAUCIE_labels(epochs=2, lr=1e-6, normalize=True,
                           random_state=42, verbose=0)
    saucie.fit(data)
    encoded = saucie.transform(data)
    assert encoded.shape == (10000, 2)


def test_SAUCIE_batches_preserves_data_shape():
    """
    test if SAUCIE cleans the data
    and does not lose the samples or features
    """
    data = data_saucie()
    batches = data_batches()
    saucie = SAUCIE_batches(epochs=2, lr=1e-9, normalize=True,
                            random_state=42, verbose=0)
    saucie.fit(data, batches)
    cleaned = saucie.transform(data, batches)
    assert cleaned.shape == (10000, 20)


def test_SAUCIE_labels_data():
    """
    test if SAUCIE clusters the data,
    decodes the labels and does not lose the samples
    """
    data = data_saucie()
    saucie = SAUCIE_labels(epochs=2, lr=1e-6, normalize=True,
                           random_state=42, verbose=0)
    saucie.fit(data)
    labels = saucie.predict(data)
    assert labels.shape == (10000, )


def test_SAUCIE_batches_preserves_ref_batch():
    """
    test if SAUCIE does not modify the samples from reference batch
    """
    data = data_saucie()
    batches = data_batches()
    saucie = SAUCIE_batches(epochs=2, lr=1e-9, normalize=False,
                            random_state=42, verbose=0)
    saucie.fit(data, batches)
    cleaned = saucie.transform(data, batches)
    ref_batch = np.where(batches == 0)
    np.testing.assert_array_equal(data[ref_batch], cleaned[ref_batch])


def test_SAUCIE_yields_stable_results_without_training():
    """
    test if SAUCIE classifier and transformer
    can be initialized with set seed and produces reproducible model
    """
    data = data_saucie()
    saucie = SAUCIE_labels(epochs=0, lr=1e-6, normalize=True,
                           random_state=42, verbose=0)
    saucie.fit(data)
    labels1 = saucie.predict(data)
    encoded1 = saucie.transform(data)

    saucie1 = SAUCIE_labels(epochs=0, lr=1e-6, normalize=True,
                            random_state=42, verbose=0)
    saucie1.fit(data)
    labels2 = saucie1.predict(data)
    encoded2 = saucie1.transform(data)
    np.testing.assert_array_equal(labels1, labels2)
    np.testing.assert_array_equal(encoded1, encoded2)


def test_SAUCIE_yields_stable_results_with_training():
    """
    test if SAUCIE classifier and transformer
    can be initialized with set seed,
    is stable during training and produces reproducible results
    """
    data = data_saucie()
    saucie = SAUCIE_labels(epochs=2, lr=1e-6, normalize=True,
                           random_state=42, verbose=0)
    saucie.fit(data)
    labels1 = saucie.predict(data)
    encoded1 = saucie.transform(data)

    saucie1 = SAUCIE_labels(epochs=2, lr=1e-6, normalize=True,
                            random_state=42, verbose=0)
    saucie1.fit(data)
    labels2 = saucie1.predict(data)
    encoded2 = saucie1.transform(data)
    np.testing.assert_array_equal(labels1, labels2)
    np.testing.assert_array_equal(encoded1, encoded2)


def test_SAUCIE_batches_yields_stable_results_without_training():
    """
    test if SAUCIE batch transformer can be initialized with set seed
    and produces reproducible model
    """
    data = data_saucie()
    batches = data_batches()
    saucie = SAUCIE_batches(epochs=0, lr=1e-9, normalize=True,
                            random_state=42, verbose=0)
    saucie.fit(data, batches)
    cleaned1 = saucie.transform(data, batches)

    saucie1 = SAUCIE_batches(epochs=0, lr=1e-9, normalize=True,
                             verbose=0, random_state=42)
    saucie1.fit(data, batches)
    cleaned2 = saucie1.transform(data, batches)

    np.testing.assert_array_equal(cleaned1, cleaned2)


def test_SAUCIE_batches_yields_stable_results_with_training():
    """
    test if SAUCIE batch transformer can be initialized with set seed,
    is stable during training and produces reproducible results
    """
    data = data_saucie()
    batches = data_batches()
    saucie = SAUCIE_batches(epochs=2, lr=1e-9, normalize=True,
                            verbose=0, random_state=42)
    saucie.fit(data, batches)
    cleaned1 = saucie.transform(data, batches)

    saucie1 = SAUCIE_batches(epochs=2, lr=1e-9, normalize=True,
                             verbose=0, random_state=42)
    saucie1.fit(data, batches)
    cleaned2 = saucie1.transform(data, batches)

    np.testing.assert_array_equal(cleaned1, cleaned2)


def test_SAUCIE_batches_yields_stable_results_batches_order():
    """
    test if SAUCIE batch transformer returns the same result for
    the nonreference batch independently of the order of transforming
    """
    data = data_saucie()
    batches = data_batches()
    saucie = SAUCIE_batches(epochs=2, lr=1e-9, normalize=True,
                            verbose=0, random_state=42)
    saucie.fit(data, batches)
    cleaned1 = saucie.transform(data, batches)

    batches[batches == 1] = 3
    # batches 2 is now before batches 1
    cleaned2 = saucie.transform(data, batches)

    np.testing.assert_array_equal(cleaned1, cleaned2)


def test_SAUCIE_is_clonable():
    """
    test if SAUCIE classifier and transformer can be cloned
    with scikit-learn clone function and retain the attributes
    """
    data = data_saucie()
    saucie = SAUCIE_labels(epochs=2, lr=1e-6, normalize=True,
                           verbose=0, random_state=42)
    saucie.fit(data)
    labels1 = saucie.predict(data)
    encoded1 = saucie.transform(data)

    saucie1 = clone(saucie)
    saucie1.fit(data)
    labels2 = saucie1.predict(data)
    encoded2 = saucie1.transform(data)
    np.testing.assert_array_equal(labels1, labels2)
    np.testing.assert_array_equal(encoded1, encoded2)


def test_SAUCIE_batches_is_clonable():
    """
    test if SAUCIE batch transformer can be cloned
    with scikit-learn clone function and retain the attributes
    """
    data = data_saucie()
    batches = data_batches()
    saucie = SAUCIE_batches(epochs=2, lr=1e-9, normalize=True,
                            verbose=0, random_state=42)
    saucie.fit(data, batches)
    cleaned1 = saucie.transform(data, batches)

    saucie1 = clone(saucie)
    saucie1.fit(data, batches)
    cleaned2 = saucie1.transform(data, batches)

    np.testing.assert_array_equal(cleaned1, cleaned2)


def test_SAUCIE_exporting_restores_tf_graph():
    """
    test if SAUCIE classifier and transformer can be exported
    with joblib, preserving the model weights
    """
    data = data_saucie()
    saucie = SAUCIE_labels(epochs=2, lr=1e-6, normalize=True, random_state=42)
    saucie.fit(data)
    labels1 = saucie.predict(data)
    encoded1 = saucie.transform(data)

    with io.BytesIO() as f:
        joblib.dump(saucie, f)
        f.seek(0)
        saucie2 = joblib.load(f)

    saucie2.fit(data)
    labels2 = saucie2.predict(data)
    encoded2 = saucie2.transform(data)
    np.testing.assert_array_equal(labels1, labels2)
    np.testing.assert_array_equal(encoded1, encoded2)


def test_SAUCIE_batches_exporting_restores_tf_graph():
    """
    test if SAUCIE batch transformer can be exported
    with joblib, preserving the model weights
    """
    data = data_saucie()
    batches = data_batches()
    saucie = SAUCIE_batches(epochs=2, lr=1e-9, normalize=True, random_state=42)
    saucie.fit(data, batches)
    cleaned1 = saucie.transform(data, batches)

    with io.BytesIO() as f:
        joblib.dump(saucie, f)
        f.seek(0)
        saucie2 = joblib.load(f)

    saucie2.fit(data, batches)
    cleaned2 = saucie2.transform(data, batches)

    np.testing.assert_array_equal(cleaned1, cleaned2)


def test_SAUCIE_labels_data_categorically():
    """
    test if SAUCIE clusters the data
    and returns consecutive integer labels
    """
    data = data_saucie()
    saucie = SAUCIE_labels(epochs=2, lr=1e-6, normalize=True,
                           random_state=42, verbose=0)
    saucie.fit(data)
    labels = saucie.predict(data)
    vals = np.unique(labels)
    expected_clusters = np.linspace(0, vals.max(), num=vals.shape[0])
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(vals, expected_clusters)
