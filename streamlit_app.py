import numpy as np
import pandas as pd
import streamlit as st

from saucie.saucie import SAUCIE_batches, SAUCIE_labels
from streamlit_elements.elements import convert_df, display_buttons
from streamlit_elements.figures import prepare_figure
from streamlit_elements.scores import display_scores

if __name__ == "__main__":
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
            normalize = st.checkbox('Normalize my data', value=True,
                                    help="""Unless you are absolutely sure
                                     your data is already normalized and is
                                      not unbalanced,
                                      do not skip normalization.""")
            submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            if batch_select == "No batches":
                model_batches = None
            else:
                batches = data[[batch_select]].to_numpy()
                batches = batches.flatten()
                data.drop(columns=batch_select, inplace=True)
                model_batches = SAUCIE_batches(epochs=2, lr=1e-9,
                                               normalize=normalize,
                                               random_state=42)

            if label_select == "No labels":
                ground_truth = None
            else:
                ground_truth = data[[label_select]].to_numpy()
                ground_truth = ground_truth.flatten()
                data.drop(columns=label_select, inplace=True)
            # keep colnames and rownames for download
            indexes = data.index.tolist()
            columns = data.columns.tolist()
            # convert data
            data = data.to_numpy()
            data = data.astype(float)

            if model_batches is not None:
                with st.spinner("Performing batch correction"):
                    # set up batch training
                    model_batches.fit(data, batches)
                    cleaned_data = model_batches.transform(data, batches)
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
            saucie = SAUCIE_labels(epochs=150, lr=1e-4, normalize=False,
                                   lambda_c=0.1, lambda_d=0.2,
                                   batch_size=256, shuffle=True)
            with st.spinner("Training the model"):
                saucie.fit(cleaned_data)
            with st.spinner("Calculating the results"):
                embedded = saucie.transform(cleaned_data)
                labels = saucie.predict(cleaned_data)
            with st.spinner("Preparing the plot"):
                fig = prepare_figure(embedded[:, 0], embedded[:, 1],
                                     labels, ground_truth)
            st.plotly_chart(fig, use_container_width=True)

            display_scores(cleaned_data, embedded, labels, ground_truth)
            labels_csv = convert_df(pd.DataFrame(labels, index=indexes,
                                                 columns=["label"]))
            embedded_csv = convert_df(pd.DataFrame(embedded, index=indexes,
                                                   columns=["SAUCIE1",
                                                            "SAUCIE2"]))
            if batched:
                cleaned_data = convert_df(pd.DataFrame(cleaned_data,
                                          index=indexes, columns=columns))
            # labels, embedding, model, cleaned data, model for batches
            display_buttons(labels_csv, embedded_csv,
                            cleaned_data, batched)
