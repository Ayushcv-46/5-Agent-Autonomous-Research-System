# app.py
# AutoResearch — Modular refactor
#
# Thin Streamlit entry point. All logic lives in sub-packages.

import streamlit as st

from ui.styling import apply_styling
from ui.sidebar import render_sidebar
from ui.chat import render_chat

# ---- Page config & CSS ----
apply_styling()

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "research_history" not in st.session_state:
    st.session_state.research_history = []

# ---- Sidebar (returns toggle values) ----
run_ragas_toggle, _ragas_available = render_sidebar()

# ---- Main chat interface ----
render_chat(run_ragas_toggle)
