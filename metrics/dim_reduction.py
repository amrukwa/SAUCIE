import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def get_optimal_label(X, y):
    labels, counts = np.unique(y, return_counts=True)
    closeness = counts - 500
    up_close = counts[closeness > 0]
    label = np.where(counts == np.min(up_close))[0][0]
    label = labels[np.argmin(closeness)]
    return label


def getVar(latent, nodes):
    d = pairwise_distances(latent[nodes, :])
    d = d[d != 0]
    var = np.std(d)**2/np.mean(d)
    return var


def get_clique_var(cliques_sub, sublatent):
    lat_vars = np.mean([getVar(sublatent, i) for i in cliques_sub])
    return lat_vars


def get_cliques(sub_dists):
    avg = np.mean(sub_dists)
    std = np.std(sub_dists)
    thresh = 2.8
    low_values = sub_dists <= (avg+std/thresh)
    high_values = sub_dists >= (avg-std/thresh)
    known_values = high_values & low_values
    adjacency_matrix = known_values.astype(int)
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    all_rows = range(0, adjacency_matrix.shape[0])
    for n in all_rows:
        gr.add_node(n)
    gr.add_edges_from(edges)
    cliques = list(nx.find_cliques(gr))
    cliquesSub = [i for i in cliques if len(i) >= 3]
    return cliquesSub


def get_variances(sub_mat, sub_latent):
    sub_dists = pairwise_distances(sub_mat)
    cliquesSub = get_cliques(sub_dists)
    amb_var = get_clique_var(cliquesSub, sub_mat)
    lat_var = get_clique_var(cliquesSub, sub_latent)
    return amb_var, lat_var


def frac_unique_neighbors(latent, cluster_label, metric=1, neighbors=30):
    """ Calculates the fraction of nearest neighbors from same cell type
    latent : numpy array of latent space (n_obs x n_latent)
    cluster_label : list of labels for all n_obs
    metrics : Distance metric, 1 = manhattan
    neighbors : No. of nearest neighbors to consider
    Returns:
    Dictionary mapping each unique label in the class cluster_label
    to list of fraction of neighbors in the same label for each cell,
    and Dictionary mapping each unique label in the category cluster_label
    to a list of unique labels of each cell's neighbors
    """
    cats = pd.Categorical(cluster_label)
    # Get nearest neighbors in each space
    n = neighbors
    neigh = NearestNeighbors(n_neighbors=n, p=metric)
    # Get transformed count matrices
    clusters = np.unique(cluster_label)
    unique_clusters = {}
    frac_neighs = np.array([])
    X_full = latent
    neigh.fit(X_full)
    for c in clusters:
        X = latent[cats == c, :]
        # Find n nearest neighbor cells (L1 distance)
        kN = neigh.kneighbors(X)
        matN = kN[1]
        frac = np.zeros(matN.shape[0])
        # How many of top n neighbors come from same cluster
        # in the labeled data (out of n neighbors)
        fr_range = range(0, len(frac))
        unique_clusters[c] = np.unique([cats[matN[i]] for i in fr_range])
        fracs = [cats[matN[i]].value_counts()[c]/n for i in fr_range]
        frac_neighs = np.append(frac_neighs, fracs)
    return frac_neighs, unique_clusters
