import numpy as np
import streamlit as st
from sklearn.metrics import adjusted_rand_score, silhouette_score


@st.cache
def compute_ari_score(y_pred, y_true=None):
    ari_score = adjusted_rand_score(y_true, y_pred)
    return ari_score


@st.cache
def compute_silhouette(x, y):
    score = silhouette_score(x, y)
    return score


def display_scores(X, X_embed, y_pred, y_true=None):
    col1, col2, col3 = st.columns(3)
    if np.unique(y_pred).shape[0] == 1:
        silhouette_labels = "-"
    else:
        silhouette_labels = compute_silhouette(X, y_pred)
        silhouette_labels = round(float(silhouette_labels), 5)
    col1.metric("Silhouette of the clustering",
                value=silhouette_labels)
    if y_true is not None:
        ari = compute_ari_score(y_pred, y_true)
        ari = round(float(ari), 5)
        silhouette_embed = compute_silhouette(X_embed, y_true)
        silhouette_embed = round(float(silhouette_embed), 5)
        col3.metric("Adjusted Rand Index",
                    value=round(ari, 5))
        col2.metric("Silhouette of the embedding",
                    value=silhouette_embed)
