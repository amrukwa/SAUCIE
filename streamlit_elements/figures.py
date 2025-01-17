import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


@st.cache
def prepare_figure(x, y, saucie_label, ground_truth=None,
                   batches=None):
    """
    Prepare figure to display in the streamlit
    """
    subplot_titles = ["Original labels"]
    if ground_truth is not None:
        subplot_titles.append("SAUCIE results")
    if batches is not None:
        subplot_titles.append("batches")

    if len(subplot_titles) > 1:
        fig = prepare_subplots(subplot_titles)
        fig = get_subplot(fig, x, y, ground_truth,
                          title="Original labels", col=1)
        if ground_truth is not None:
            fig = get_subplot(fig, x, y, saucie_label,
                              title="SAUCIE labels", col=2)
        if batches is not None:
            fig = get_subplot(fig, x, y, batches,
                              title="batches",
                              col=len(subplot_titles))
    else:
        fig = prepare_single_plot()
        fig = add_to_single_plot(fig, x, y, saucie_label)
    return fig


def prepare_subplots(subplot_titles):
    """
    Prepare subplots for the figure
    """
    fig = make_subplots(rows=1, cols=len(subplot_titles),
                        shared_xaxes=True,
                        subplot_titles=subplot_titles,
                        shared_yaxes=True)
    fig.update_xaxes(title_text='SAUCIE 1')
    fig.update_yaxes(title_text='SAUCIE 2')
    fig.update_xaxes(matches='x')
    fig.update_layout(legend=dict(groupclick="toggleitem"))
    return fig


def prepare_single_plot():
    """
    Prepare the layout for a single plot
    """
    fig = go.Figure()
    fig.update_layout(title_text="SAUCIE results")
    fig.update_xaxes(title_text='SAUCIE 1')
    fig.update_yaxes(title_text='SAUCIE 2')
    return fig


def add_to_single_plot(fig, x, y, labels):
    """
    Add the data to a single plot
    """
    single_labels = np.unique(labels)
    for label in single_labels:
        fig.add_trace(go.Scatter(x=x[np.where(labels == label)],
                                 name=str(label),
                                 showlegend=True,
                                 y=y[np.where(labels == label)],
                                 mode='markers'))
    return fig


def get_subplot(fig, x, y, labels, title="", col=1):
    """
    Add the data to a specified subplot
    """
    single_labels = np.unique(labels)
    for label in single_labels:
        fig.add_trace(go.Scatter(x=x[np.where(labels == label)],
                                 name=str(label),
                                 showlegend=True,
                                 legendgroup=str(col),
                                 legendgrouptitle_text=title,
                                 y=y[np.where(labels == label)],
                                 mode='markers'),
                      row=1, col=col)
    return fig
