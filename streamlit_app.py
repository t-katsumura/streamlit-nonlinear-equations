import streamlit as st

import lorentz_attractor

PAGES = {
    "Lorentz Attractor": lorentz_attractor
}

st.sidebar.title('Non Linear Equations')
selection = st.sidebar.radio("Select equation", list(PAGES.keys()))
page = PAGES[selection]
page.app()