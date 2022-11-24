import numpy as np
import scipy.spatial.distance as ssdist


def _generate_centroids(x, labels):
    centroids = np.zeros((np.unique(labels).shape[0], x.shape[1]))
    for i in range(np.unique(labels).shape[0]):
        centroids[i, :] = np.mean(x[np.where(labels == i)], axis=0)
    return centroids


def dunn_index(x, labels, metric, centroids=None):
    if centroids is None:
        centroids = _generate_centroids(x, labels)
    datasize = x.shape[0]
    n_clusters = centroids.shape[0]
    intercluster_distance = ssdist.cdist(centroids, centroids, metric)
    min_ctc = intercluster_distance[1, 0]
    for i in range(n_clusters):
        for j in range(n_clusters):
            if min_ctc > intercluster_distance[i, j] > 0:
                min_ctc = intercluster_distance[i, j]
    distance_ctp = ssdist.cdist(x, centroids, metric)
    intracluster_distance = [distance_ctp[labels[i]] for i in range(datasize)]
    max_intradist = np.max(intracluster_distance)
    return min_ctc / max_intradist
