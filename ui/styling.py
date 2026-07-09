# ui/styling.py
# Streamlit page config and custom CSS.

import streamlit as st


def apply_styling():
    st.set_page_config(page_title="AutoResearch", layout="centered")

    st.markdown(
        """
        <style>
            .block-container { max-width: 760px; padding-top: 2.5rem; padding-bottom: 6rem; }
            [data-testid="stChatMessage"] { padding: 0.35rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AutoResearch")
    st.caption("Five agents plan, search, read, write, and critique — now RAGAS-evaluated.")
