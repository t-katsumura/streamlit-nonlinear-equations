import streamlit as st

import lorentz_attractor
import chua_circuit

PAGES = {
    "Lorentz Attractor": lorentz_attractor,
    "Chua's Circuit": chua_circuit,
}

st.sidebar.title('Non Linear Equations')
selection = st.sidebar.radio("Select equation", list(PAGES.keys()))
page = PAGES[selection]
page.app()