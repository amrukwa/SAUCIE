import streamlit as st

from streamlit_elements.download_button import download_button


def display_buttons(labels, dim_red,
                    model, cleaned=None, model_batch=None):
    """
    display the buttons in the app window
    """
    col1, col2 = st.columns([0.4, 1], gap="small")

    with col1:
        download_labels = download_button(labels, "labels.csv",
                                          'Download labels',
                                          pickle_it=False)
        st.markdown(download_labels, unsafe_allow_html=True)
        download_dim_red = download_button(dim_red, "reduced_data.csv",
                                           'Download embedding',
                                           pickle_it=False)
        st.markdown(download_dim_red, unsafe_allow_html=True)
        if model_batch is not None:
            download_cleaned = download_button(cleaned, "cleaned_data.csv",
                                               'Download cleaned data',
                                               pickle_it=False)
            st.markdown(download_cleaned, unsafe_allow_html=True)
    with col2:
        download_model = download_button(model, "model.pickle",
                                         'Download model',
                                         pickle_it=True)
        st.markdown(download_model, unsafe_allow_html=True)
        if model_batch is not None:
            download_model_b = download_button(model_batch,
                                               "model_batches.pickle",
                                               'Download model  - batches',
                                               pickle_it=True)
            st.markdown(download_model_b, unsafe_allow_html=True)


@st.cache
def convert_df(df):
    """
    convert dataframe to downloading format
    """
    return df.to_csv().encode('utf-8')
