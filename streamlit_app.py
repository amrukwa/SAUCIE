import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.callbacks import Callback, EarlyStopping

from saucie.saucie import SAUCIE_batches, SAUCIE_labels
from streamlit_elements.elements import convert_df, display_buttons
from streamlit_elements.figures import prepare_figure
from streamlit_elements.prepare_data import extract_metalabel, filter_data
from streamlit_elements.scores import display_scores


class streamlitCallbacks(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global progress
        global progress_bar
        global epochs
        global batch_no
        progress += 1
        progress_bar.progress(int(progress*100/(epochs*batch_no)))


if __name__ == "__main__":

    epochs = 100
    progress = 0
    batch_no = 1

    st.set_page_config(
        page_title="SAUCIE", page_icon="🔎"
        )

    st.write("""
        # SAUCIE
        """)

    st.markdown("### Upload your data file")
    uploaded = st.file_uploader("your data file", type=[".csv"],
                                accept_multiple_files=False,
                                help="help here", on_change=None,
                                args=None, kwargs=None,
                                disabled=False, label_visibility="hidden")
    if uploaded:

        data = pd.read_csv(uploaded, index_col=0)

        st.markdown("### Select original labels and batches")
        with st.form(key="my_form"):
            label_select = st.selectbox(
                "Label",
                options=["No labels"]+data.columns.tolist(),
                help="""
                Select which column refers to your labels.
                If none, select "No labels" and submit.
                """,
            )

            batch_select = st.selectbox(
                    "Batch",
                    options=["No batches"]+data.columns.tolist(),
                    help="""
                    Select which column refers to your batches.
                    If none, select "No batches" and submit.
                    """,
                )

            frac = st.slider("""What part of genes with highest
                             variances should
                             remain after filtering""",
                             0.1, 1.0, 1.0,
                             step=0.05,
                             help="""The genes with zeros only will be
                             removed before filtering.""")
            normalize = st.checkbox('Normalize my data', value=True,
                                    help="""Unless you are absolutely sure
                                     your data is already normalized and is
                                      not unbalanced,
                                      do not skip normalization.""")
            b_download = st.checkbox("""Prepare batch corrected data for
                                    download""",
                                     value=False,
                                     help="""Converting the data to .csv
                                     may take some
                                     time and make the application
                                     much slower.
                                     Applies only for data with batches.""")
            submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            progress_info = st.empty()
            progress_bar = st.progress(progress)
            st_callback = streamlitCallbacks()

            if batch_select == "No batches":
                model_batches = None
                batches = None
            else:
                epochs = 500
                callbacks = [st_callback, EarlyStopping(monitor='mmd_loss',
                             patience=25, restore_best_weights=True)]
                data, batches = extract_metalabel(data, batch_select)
                model_batches = SAUCIE_batches(lambda_b=2.5,
                                               batch_size=1024,
                                               epochs=epochs,
                                               lr=2e-3,
                                               normalize=normalize,
                                               random_state=123,
                                               callback=callbacks)

            if label_select == "No labels":
                ground_truth = None
            else:
                data, ground_truth = extract_metalabel(data, label_select)

            # keep rownames for download
            indexes = data.index.tolist()
            # convert and filter data
            data, columns = filter_data(data, frac=frac)

            if model_batches is not None:
                with progress_info:
                    # set up batch training
                    batch_no = np.unique(batches).shape[0] - 1
                    st.info(("Performing batch correction"))
                    model_batches.fit(data, batches)
                    cleaned_data = model_batches.transform(data, batches)
                    batch_no = 1
                    progress = 0
                    epochs = 100
                    del model_batches
                batched = True
            else:
                batched = False
                if normalize:
                    data = (data - np.min(data, axis=0))
                    data = data/np.max(data, axis=0)
                    cleaned_data = np.arcsinh(data)
                else:
                    cleaned_data = data

            # fit on the cleaned data -> labels, embed
            callbacks = [st_callback, EarlyStopping(monitor='loss',
                         patience=20, restore_best_weights=True)]
            saucie = SAUCIE_labels(epochs=epochs, lr=1e-4, normalize=False,
                                   lambda_c=0.1, lambda_d=0.4,
                                   layers=[512, 256, 128, 2],
                                   batch_size=256, shuffle=True,
                                   callback=callbacks)

            with progress_info:
                st.info("Training the model")
                saucie.fit(cleaned_data)
            with progress_info:
                st.info("Calculating the results")
                progress_bar.progress(0)
                embedded = saucie.transform(cleaned_data)
                progress_bar.progress(50)
                labels = saucie.predict(cleaned_data)
                progress_bar.progress(100)
            with progress_info:
                st.info("Preparing the plot")
                fig = prepare_figure(embedded[:, 0], embedded[:, 1],
                                     labels, ground_truth, batches)
            st.plotly_chart(fig, use_container_width=True)
            with progress_info:
                st.info("Calculating scores")
            display_scores(cleaned_data, embedded, labels, ground_truth)
            with progress_info:
                st.info("Preparing download files")
            labels_csv = convert_df(pd.DataFrame(labels, index=indexes,
                                                 columns=["label"]))
            embedded_csv = convert_df(pd.DataFrame(embedded, index=indexes,
                                                   columns=["SAUCIE1",
                                                            "SAUCIE2"]))
            del saucie
            if batched and b_download:
                cleaned_data = cleaned_data - cleaned_data.min()
                cleaned_data = cleaned_data.astype(np.float16)
                cleaned_data = convert_df(pd.DataFrame(cleaned_data,
                                          index=indexes, columns=columns))
            # labels, embedding, model, cleaned data, model for batches
            display_buttons(labels_csv, embedded_csv,
                            cleaned_data, batched and b_download)
            with progress_info:
                st.info("Done!")
