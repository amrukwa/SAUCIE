import io

import joblib
import streamlit as st


def display_buttons(labels, dim_red,
                    model, cleaned=None, model_batch=None):
    """
    display the buttons in the app window
    """
    col1, col2 = st.columns([0.4, 1], gap="small")

    col1.download_button("Download labels ",
                         labels,
                         file_name="labels.csv",
                         mime='text/csv')

    col2.download_button("Download model",
                         model,
                         file_name="model.joblib",
                         help="""Download model for clustering
                          and dimensionality reduction""",
                         mime='application/octet-stream')

    col1.download_button("Download embedding",
                         dim_red,
                         file_name="reduced_data.csv",
                         mime='text/csv')

    if model_batch is not None:
        col2.download_button("Download model - batches",
                             model_batch,
                             file_name="model_batches.joblib",
                             help="""Download model for batch correction
                              and data cleaning""",
                             mime='application/octet-stream')

        col1.download_button("Download cleaned data",
                             cleaned,
                             file_name="cleaned_data.csv",
                             mime='text/csv')


@st.cache
def dump_model(model):
    """
    export joblib model for downloading
    """
    f = io.BytesIO()
    joblib.dump(model, f)
    f.seek(0)
    return f


@st.cache
def convert_df(df):
    """
    convert dataframe to downloading format
    """
    return df.to_csv().encode('utf-8')
