import argparse
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import f_oneway, tukey_hsd


def path(folder):
    if os.path.exists(folder) and os.path.isdir(folder):
        return folder
    raise ValueError(f"Folder not found or not a folder: {folder}")


def common_layout_update(fig, height, width, t):
    fig.update_layout(
        height=height, width=width,
        margin=dict(l=5, r=5, b=5, t=t),
        template="simple_white",
        font_family="Computer Modern",
        title_font_family="Computer Modern")
    return fig


def plot_mixing_ratios(files, folder=""):
    np.random.seed(1)
    mixes = ["original", "SAUCIE",
             "PCA", "PCA + UMAP", "PCA + t-SNE"]
    subfolder = "ratios/"
    fig = go.Figure()
    for i, name in enumerate(mixes):
        ratios = np.genfromtxt(folder+subfolder+files[i]+".csv", delimiter=',')
        fig.add_trace(go.Box(y=ratios, name=name))
    fig = common_layout_update(fig, 400, 700, t=5)
    return fig


def get_subplot(fig, x, y, labels,
                col=1, row=1, showlegend=False,
                main_col=True):
    """
    Add the data to a specified subplot
    """
    single_labels = np.unique(labels)
    if main_col:
        colors = px.colors.qualitative.Plotly
    else:
        colors = px.colors.qualitative.Safe
        if single_labels.shape[0] > 10:
            colors = px.colors.qualitative.Alphabet
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
                          "DiviK", "Louvain",
                          "k-means", "hierarchical"]
    embed = pd.read_csv(folder+"embed/"+embed_file+".csv",
                        header=None)
    embed = embed.to_numpy().T
    subfolder = "cluster/"

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

    fig = common_layout_update(fig, 1000, 1200, t=10)
    fig.update_layout(legend=dict(yanchor="top",
                                  y=1.025,
                                  xanchor="left",
                                  x=0.37))
    fig = common_layout_update(fig, 1000, 1200, t=10)
    return fig


def plot_dim_red(cluster_file, files, folder=""):
    dim_red_methods = ["SAUCIE", "PCA",
                       "PCA + UMAP",
                       "PCA + t-SNE"]
    nrows = 2
    ncols = 2
    labels = pd.read_csv(folder+"cluster/"+cluster_file+".csv",
                         delimiter=',', header=None)
    subfolder = "embed/"

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
    fig = common_layout_update(fig, 1000, 1200, t=30)
    # fig.update_layout(legend=dict(yanchor="top",
    #                               y=0.25,
    #                               xanchor="left",
    #                               x=0.55))
    return fig


def compare_distributions(files, folder=""):
    subfolder = "ratios/"
    original = np.genfromtxt(folder+subfolder+files[0]+".csv", delimiter=',')
    saucie = np.genfromtxt(folder+subfolder+files[1]+".csv", delimiter=',')
    pca = np.genfromtxt(folder+subfolder+files[2]+".csv", delimiter=',')
    umap = np.genfromtxt(folder+subfolder+files[3]+".csv", delimiter=',')
    tsne = np.genfromtxt(folder+subfolder+files[4]+".csv", delimiter=',')
    pval = f_oneway(original, saucie, pca, umap, tsne).pvalue
    print('{0:.16f}'.format(pval))
    if pval < 0.05:
        print(tukey_hsd(original, saucie, pca, umap, tsne))


def plot_batch_correction(batch_file, files, folder):
    dim_red_methods = ["SAUCIE batch uncorrected", "SAUCIE batch corrected",
                       "PCA", "PCA + UMAP", "PCA + t-SNE"]
    batch = pd.read_csv(folder+batch_file+".csv",
                        delimiter=',', header=None)
    subfolder = "embed/"
    nrows = 3
    ncols = 2
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=dim_red_methods)
    for i in range(1, nrows+1):
        for j in range(1, ncols+1):
            if (i-1)*ncols+j-1 == 5:
                break
            filename = folder+subfolder+files[(i-1)*ncols+j-1]+".csv"
            embed = pd.read_csv(filename,
                                delimiter=',',
                                header=None)
            embed = embed.to_numpy().T
            showlegend = i == 1 and j == 1
            fig = get_subplot(fig, embed[0, :], embed[1, :], batch,
                              col=j, row=i, showlegend=showlegend)
    fig = common_layout_update(fig, 1000, 1200, t=20)
    # fig.update_layout(legend=dict(yanchor="top",
    #                               y=1.025,
    #                               xanchor="left",
    #                               x=0.37))
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', type=path, default="")
    parser.add_argument('--plotfolder', type=path, default="")
    parser.add_argument('--batches', action='store_true')
    parser.add_argument('--no-batches', dest='batches', action='store_false')
    parser.set_defaults(batches=False)
    args = parser.parse_args()

    files_ratio = ["original", "saucie",
                   "pca", "umap", "tsne"]
    files_cluster = ["original", "saucie",
                     "divik", "louvain",
                     "kmeans", "hierarchical"]
    files_dim_red = ["saucie", "pca",
                     "umap", "tsne"]
    files_batches = ["saucie0", "saucie",
                     "pca", "umap", "tsne"]

    compare_distributions(files_ratio, args.datafolder)
    fig = plot_mixing_ratios(files_ratio, args.datafolder)
    fig.write_image(args.plotfolder+"mixing_ratios_boxplot.png")

    fig = plot_clustering(files_dim_red[0], files_cluster, args.datafolder)
    fig.write_image(args.plotfolder+"cluster_scatter.png")

    fig = plot_dim_red(files_cluster[0], files_dim_red, args.datafolder)
    fig.write_image(args.plotfolder+"dim_red_scatter.png")
    if args.batches:
        fig = plot_batch_correction("batches", files_batches, args.datafolder)
        fig.write_image(args.plotfolder+"batch_scatter.png")
