import streamlit as st
import subprocess
import json
import sys
from pathlib import Path


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Neural Image Retrieval",
    page_icon="🔎",
    layout="wide"
)


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------------

st.title("🔎 Neural Caption-Based Image Retrieval")

model_options = {
    "Original GRU + GloVe": ("config_author.yaml", "gru_glove", "GRU + GloVe", "1,024")
}
improvement_report = PROJECT_ROOT / "outputs" / "improved" / "selected_comparison.json"
improvement_checkpoint = PROJECT_ROOT / "checkpoints" / "improved" / "ensemble_glove_best.pt"
if improvement_report.is_file() and improvement_checkpoint.is_file():
    model_options = {
        "Improved combined model": ("config_improved.yaml", "ensemble_glove", "GRU + BiGRU + GloVe", "2,048"),
        **model_options
    }
selected_model = st.selectbox("Retrieval model", list(model_options))
model_config, model_method, model_label, model_dimensions = model_options[selected_model]

st.markdown(
    f"""
    **Search a COCO image gallery using natural language.**

    Enter a description of the image you are looking for and
    the trained **{model_label}** model will rank the gallery images
    by how closely they match your description.
    """
)


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

st.subheader("Enter Image Description")

query = st.text_input(
    "Describe the image you want to find:",
    placeholder="Example: a guy riding a bike next to a train"
)


top_k = st.selectbox(
    "Number of results:",
    [3, 5, 10],
    index=1
)


# ---------------------------------------------------------
# SEARCH BUTTON
# ---------------------------------------------------------

if st.button("🔍 Search Images", use_container_width=True):

    if not query.strip():
        st.warning("Please enter an image description.")

    elif not any(character.isalnum() for character in query):
        st.warning("Please enter an image description containing words.")

    else:

        if len(query.split()) > 50:
            st.warning("Only the first 50 words are used for this search.")

        with st.spinner("Searching the image gallery..."):

            command = [
    sys.executable,
    "-m",
    "src.retrieve",
    "--config",
    model_config,
    "--method",
    model_method,
    "--query",
    query,
    "--top-k",
    str(top_k)
]

            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True
            )

        # -------------------------------------------------
        # HANDLE ERRORS
        # -------------------------------------------------

        if result.returncode != 0:

            if "Query contains no known words" in result.stderr:
                st.warning("The model could not recognize any words in this description. Try a simple English description.")
            else:
                st.error("An error occurred while running the model.")
                st.code(
                    result.stderr,
                    language="text"
                )

        else:

            # -------------------------------------------------
            # READ MODEL OUTPUT
            # -------------------------------------------------

            try:

                results = json.loads(result.stdout)

                st.success(
                    f"Retrieved {len(results)} images successfully."
                )

                st.divider()

                st.subheader("Retrieval Results")

                # -------------------------------------------------
                # DISPLAY RESULTS
                # -------------------------------------------------

                columns = st.columns(3)

                for index, item in enumerate(results):

                    column = columns[index % 3]

                    with column:

                        image_path = Path(item["path"])

                        st.markdown(
                            f"### Rank #{item['rank']}"
                        )

                        if image_path.exists():

                            st.image(
                                str(image_path),
                                use_container_width=True
                            )

                        else:

                            st.warning(
                                "Image file not found."
                            )

                        st.metric(
                            "Similarity Score",
                            f"{item['score']:.4f}"
                        )

                        st.caption(
                            item["caption"]
                        )

                        st.divider()

            except json.JSONDecodeError:

                st.error(
                    "The model returned an unexpected output."
                )

                st.code(
                    result.stdout,
                    language="text"
                )


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

st.divider()

st.subheader("Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", model_label)

with col2:
    st.metric("Gallery", "1,000 Images")

with col3:
    st.metric("Embedding", model_dimensions)

with col4:
    st.metric("Dataset", "COCO 2014")
