"""Streamlit entry point for the classification comparison project."""

import streamlit as st


def main() -> None:
    """Display the initial project landing page."""
    st.set_page_config(
        page_title="Classification Model Comparison",
        page_icon="📊",
        layout="wide",
    )
    st.title("Classification Model Comparison")
    st.write(
        "Compare five classification models using a shared dataset and "
        "consistent evaluation metrics."
    )

    models_column, metrics_column = st.columns(2)
    with models_column:
        st.subheader("Models")
        st.markdown(
            "- Logistic Regression\n"
            "- Decision Tree\n"
            "- K-Nearest Neighbors\n"
            "- Naive Bayes\n"
            "- Support Vector Machine"
        )
    with metrics_column:
        st.subheader("Evaluation metrics")
        st.markdown(
            "- Accuracy\n"
            "- AUC\n"
            "- Precision\n"
            "- Recall\n"
            "- F1\n"
            "- Matthews Correlation Coefficient (MCC)"
        )

    st.info("No evaluation results are available yet.")


if __name__ == "__main__":
    main()
