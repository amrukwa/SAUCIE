import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def path(folder):
    if os.path.exists(folder) and os.path.isdir(folder):
        return folder
    raise ValueError(f"Folder not found or not a folder: {folder}")


def plot_mixing_ratios(files, folder=""):
    np.random.seed(1)
    mixes = ["original", "SAUCIE", "DiVAE",
             "PCA", "PCA + UMAP", "PCA + t-SNE"]
    subfolder = "data_plots/ratios/"
    fig = go.Figure()
    for i, name in enumerate(mixes):
        ratios = np.genfromtxt(folder+subfolder+files[i]+".csv", delimiter=',')
        fig.add_trace(go.Box(y=ratios, name=name))
    fig.update_layout(height=400, width=700,
                      margin=dict(l=5, r=5, b=5, t=5),
                      showlegend=False,
                      template="simple_white",
                      font_family="Computer Modern",
                      title_font_family="Computer Modern")
    fig.write_image(folder+"images/mixing_ratios_boxplot.png")


def get_subplot(fig, x, y, labels,
                col=1, row=1, showlegend=False,
                main_col=True):
    """
    Add the data to a specified subplot
    """
    if main_col:
        colors = px.colors.qualitative.Plotly
    else:
        colors = px.colors.qualitative.Safe
    single_labels = np.unique(labels)
    for i, label in enumerate(single_labels):
        fig.add_trace(go.Scatter(x=x[np.where(labels == label)[0]],
                                 name=str(label),
                                 showlegend=showlegend,
                                 marker=dict(color=colors[i]),
                                 y=y[np.where(labels == label)[0]],
                                 mode='markers'),
                      row=row, col=col)
    return fig


def plot_clustering(embed_file, files, folder=""):
    nrows = 3
    ncols = 2
    clustering_methods = ["Ground truth", "SAUCIE",
                          "DiVAE", "Louvain",
                          "k-means", "hierarchical"]
    embed = pd.read_csv(folder+"data_plots/embed/"+embed_file+".csv",
                        header=None)
    embed = embed.to_numpy().T
    subfolder = "data_plots/cluster/"

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=clustering_methods,
                        horizontal_spacing=0.25)
    fig.update_xaxes(title_text='SAUCIE 1')
    fig.update_yaxes(title_text='SAUCIE 2')

    for i in range(1, nrows+1):
        for j in range(1, ncols+1):
            filename = folder+subfolder+files[(i-1)*ncols+j-1]+".csv"
            labels = pd.read_csv(filename,
                                 delimiter=',', header=None)
            showlegend = i == 1 and j == 1
            fig = get_subplot(fig, embed[0, :], embed[1, :], labels,
                              col=j, row=i, showlegend=showlegend,
                              main_col=showlegend)
    fig.update_layout(
        height=1000, width=1200,
        margin=dict(l=5, r=5, b=5, t=10),
        template="simple_white",
        font_family="Computer Modern",
        title_font_family="Computer Modern")
    fig.update_layout(legend=dict(yanchor="top",
                                  y=1.025,
                                  xanchor="left",
                                  x=0.37))

    fig.write_image(folder+"images/cluster_scatter.png")


def plot_dim_red(cluster_file, files, folder=""):
    dim_red_methods = ["SAUCIE", "DiVAE",
                       "PCA", "PCA + UMAP",
                       "PCA + t-SNE"]
    nrows = 3
    ncols = 2
    labels = pd.read_csv(folder+"data_plots/cluster/"+cluster_file+".csv",
                         delimiter=',', header=None)
    subfolder = "data_plots/embed/"

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=dim_red_methods)
    for i in range(1, nrows+1):
        for j in range(1, ncols+1):
            if i == 3 and j == 2:
                break
            filename = folder+subfolder+files[(i-1)*ncols+j-1]+".csv"
            embed = pd.read_csv(filename,
                                delimiter=',', header=None)
            embed = embed.to_numpy().T
            showlegend = i == 1 and j == 1
            fig = get_subplot(fig, embed[0, :], embed[1, :], labels,
                              col=j, row=i, showlegend=showlegend)
    fig.update_layout(
        height=1000, width=1200,
        margin=dict(l=5, r=5, b=5, t=30),
        template="simple_white",
        font_family="Computer Modern",
        title_font_family="Computer Modern")
    fig.update_layout(legend=dict(yanchor="top",
                                  y=0.25,
                                  xanchor="left",
                                  x=0.55))
    fig.write_image(folder+"images/dim_red_scatter.png")


def compare_distributions(files, folder=""):
    print(files)
    subfolder = "data_plots/ratios/"
    original = np.genfromtxt(folder+subfolder+files[0]+".csv", delimiter=',')
    saucie = np.genfromtxt(folder+subfolder+files[1]+".csv", delimiter=',')
    divae = np.genfromtxt(folder+subfolder+files[2]+".csv", delimiter=',')
    pca = np.genfromtxt(folder+subfolder+files[3]+".csv", delimiter=',')
    umap = np.genfromtxt(folder+subfolder+files[4]+".csv", delimiter=',')
    tsne = np.genfromtxt(folder+subfolder+files[5]+".csv", delimiter=',')
    pval = f_oneway(original, saucie, divae, pca, umap, tsne).pvalue
    print('{0:.16f}'.format(pval))
    if pval < 0.05:
        print(tukey_hsd(original, saucie, divae, pca, umap, tsne))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=path, default="")
    files_ratio = ["original", "saucie", "divae",
                   "pca", "umap", "tsne"]
    files_cluster = ["original", "saucie",
                     "divae", "louvain",
                     "kmeans", "hierarchical"]
    files_dim_red = ["saucie", "divae",
                     "pca", "umap",
                     "tsne"]
    args = parser.parse_args()
    folder = args.folder
    compare_distributions(files_ratio, folder)
    plot_mixing_ratios(files_ratio, folder)
    plot_clustering(files_dim_red[0], files_cluster, folder)
    plot_dim_red(files_cluster[0], files_dim_red, folder)
