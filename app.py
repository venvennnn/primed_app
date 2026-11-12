"""Primed entry point.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from primed.ui.main import main

st.set_page_config(page_title="Primed", layout="wide", initial_sidebar_state="collapsed")
main()
